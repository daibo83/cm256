# Reliable Transport Architecture

## Overview

A reliable transport layer for cm256 that combines:
- **Streaming FEC** (existing cm256) for instant recovery of most losses
- **Sliding window ARQ** with 64-symbol bitmap ACKs for residual losses
- **Adaptive FEC** that adjusts redundancy based on observed loss
- **BBR congestion control** for bandwidth estimation and pacing

**Target latency**: 50ms + RTT

---

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      ReliableTransport                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  Streaming   │   │   Adaptive   │   │     BBR      │            │
│  │  FEC         │   │   FEC Tuner  │   │     CC       │            │
│  │  (existing)  │   │   (new)      │   │    (new)     │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│                   ┌────────▼────────┐                               │
│                   │   ARQ Layer     │                               │
│                   │  (new)          │                               │
│                   │  - Send buffer  │                               │
│                   │  - Retransmit   │                               │
│                   │  - ACK/bitmap   │                               │
│                   └────────┬────────┘                               │
│                            │                                        │
│                   ┌────────▼────────┐                               │
│                   │   Transport     │                               │
│                   │  (UDP/QUIC)     │                               │
│                   └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ARQ granularity | Symbol-level | Matches streaming FEC, minimal retransmit overhead |
| Window type | Sliding window | Continuous flow, matches diagonal interleaving |
| ACK format | 64-symbol bitmap | 8 bytes, covers ~128ms at typical rates |
| Retransmit strategy | Source symbols | Window-independent, always valid |
| FEC adaptation | Adjust num_parities | From nyxpsi's NetworkStats concept |
| Congestion control | BBR | Bandwidth estimation + pacing |

---

## Packet Types

```rust
#[repr(u8)]
pub enum PacketType {
    Source = 0,   // Existing: data symbol
    Parity = 1,   // Existing: FEC repair symbol  
    Ack = 2,      // NEW: bitmap acknowledgment
}
```

---

## ACK Packet Format (16 bytes)

```text
┌─────────┬─────────┬─────────┬─────────┬─────────┬───────────┐
│ base_seq│ type=2  │ max_seen│ loss_%  │ rtt_ms  │ reserved  │
│ (2B)    │ (1B)    │ (2B)    │ (1B)    │ (1B)    │ (1B)      │
├─────────┴─────────┴─────────┴─────────┴─────────┴───────────┤
│              Bitmap (8 bytes = 64 bits)                     │
│         bit i = 1 if (base_seq + i) received                │
└─────────────────────────────────────────────────────────────┘
```

### Field Definitions

| Field | Size | Description |
|-------|------|-------------|
| `base_seq` | 2B | Cumulative ACK: all seqs < base_seq are received |
| `type` | 1B | PacketType::Ack = 2 |
| `max_seen` | 2B | Highest seq number received (helps sender know what's in flight) |
| `loss_rate` | 1B | Observed loss % (0-255 maps to 0-100%) for adaptive FEC |
| `rtt_ms` | 1B | RTT sample in milliseconds (0-255ms) for BBR |
| `reserved` | 1B | Padding / future use |
| `bitmap` | 8B | Bit i = 1 means (base_seq + i) received |

### Bitmap Interpretation

```text
base_seq = 100
bitmap = [0b10110001, 0b11111111, ...]
         │││││││└─ seq 100: ✓ received
         ││││││└── seq 101: ✗ missing
         │││││└─── seq 102: ✗ missing
         ││││└──── seq 103: ✗ missing
         │││└───── seq 104: ✓ received
         ││└────── seq 105: ✓ received
         │└─────── seq 106: ✗ missing
         └──────── seq 107: ✓ received

Sender retransmits: 101, 102, 103, 106
```

---

## Configuration

```rust
pub struct ReliableConfig {
    // ACK parameters (for 50ms + RTT target)
    pub ack_every_n_packets: u16,    // 8
    pub ack_interval_ms: u16,         // 20
    pub min_ack_interval_ms: u16,     // 5
    pub window_size: u16,             // 64 symbols
    
    // FEC parameters (aggressive for low latency)
    pub fec_delay: u8,                // 8 symbols
    pub fec_parities: u8,             // 2-4 (adaptive)
    pub fec_step_size: u8,            // 4
    
    // ARQ parameters
    pub send_buffer_size: u16,        // 128 packets
    pub max_retries: u8,              // 2
    
    // BBR parameters
    pub initial_cwnd: u32,            // 10 packets
    pub min_rtt_probe_interval_ms: u32, // 10000 (10s)
}
```

### Presets

| Preset | ack_every_n | ack_interval_ms | fec_parities | Use Case |
|--------|-------------|-----------------|--------------|----------|
| `low_latency` | 8 | 20 | 2-4 adaptive | Real-time video, VoIP |
| `balanced` | 16 | 50 | 2-3 adaptive | Live streaming |
| `high_throughput` | 32 | 100 | 1-2 adaptive | File transfer |

---

## Components

### 1. NetworkStats (from nyxpsi)
**File**: `stats.rs`

Tracks network conditions using exponential weighted moving average:
- Packet loss rate
- RTT samples
- Network quality score (0.0 - 1.0)

```rust
pub struct NetworkStats {
    packet_loss_rate: f64,      // EWMA of loss
    latencies: VecDeque<u32>,   // Recent RTT samples
    delivered: u64,             // Bytes delivered (for BBR)
}
```

### 2. ARQ State Machine
**File**: `arq.rs`

Sender side:
- Buffers source symbols for retransmission
- Processes ACK bitmaps to identify missing symbols
- Returns list of sequences to retransmit

Receiver side:
- Tracks received symbols in 64-bit bitmap
- Builds ACK packets with current state
- Manages window advancement

```rust
pub struct SenderArq {
    send_buffer: VecDeque<SentPacket>,
    base_seq: u16,
    next_seq: u16,
}

pub struct ReceiverArq {
    base_seq: u16,
    received_bitmap: [u8; 8],  // 64 symbols
    max_seen: u16,
}
```

### 3. Adaptive FEC Tuner
**File**: `adaptive.rs`

Adjusts FEC parameters based on observed loss:

| Loss Rate | num_parities | Overhead |
|-----------|--------------|----------|
| < 5% | 1 | 12.5% |
| 5-15% | 2 | 50% |
| 15-30% | 3 | 75% |
| > 30% | 4 | 100% |

```rust
pub struct AdaptiveFec {
    current_parities: u8,
    stats: NetworkStats,
}
```

### 4. BBR Congestion Control
**File**: `bbr.rs`

Google's BBR algorithm:
- Estimates bottleneck bandwidth (btl_bw)
- Tracks minimum RTT (rt_prop)
- Controls pacing rate and congestion window

```rust
pub struct BbrState {
    btl_bw: u64,           // Bottleneck bandwidth (bytes/sec)
    rt_prop: Duration,      // Minimum RTT observed
    pacing_rate: u64,       // Current send rate
    cwnd: u32,              // Congestion window (packets)
    mode: BbrMode,          // Startup, Drain, ProbeBW, ProbeRTT
}
```

### 5. ReliableEncoder (Sender)
**File**: `encoder.rs`

Combines all components for sending:

```rust
pub struct ReliableEncoder<T> {
    fec: StreamingEncoder,
    arq: SenderArq,
    bbr: BbrState,
    adaptive: AdaptiveFec,
    transport: T,
}
```

### 6. ReliableDecoder (Receiver)
**File**: `decoder.rs`

Combines all components for receiving:

```rust
pub struct ReliableDecoder<T> {
    fec: StreamingDecoder,
    arq: ReceiverArq,
    stats: NetworkStats,
    transport: T,
}
```

---

## Data Flow

### Sender Flow

```text
User data
    │
    ▼
┌─────────────────┐
│ BBR: can_send?  │──No──→ Queue/Wait
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ StreamingEncoder│ → Generate source + parity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SenderArq       │ → Buffer source for retransmit
└────────┬────────┘
         │
         ▼
    Send packets
         │
         ▼
┌─────────────────┐
│ On ACK received │
├─────────────────┤
│ 1. Update BBR   │
│ 2. Update stats │
│ 3. Adaptive FEC │
│ 4. Retransmit?  │
└─────────────────┘
```

### Receiver Flow

```text
Receive packet
    │
    ▼
┌─────────────────┐
│ Parse header    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 Source    Parity
    │         │
    ▼         ▼
┌─────────────────┐
│ StreamingDecoder│ → FEC recovery
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ReceiverArq     │ → Update bitmap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NetworkStats    │ → Track loss/RTT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Send ACK?       │──Yes──→ Build & send ACK
└────────┬────────┘
         │ No
         ▼
    Return data
```

---

## Latency Analysis

### Target: 50ms + RTT

### Case 1: FEC Recovers (95%+ of losses)

```text
Packet lost at T=0
Parity arrives at T=10ms
FEC recovers immediately

Total latency: ~10-20ms ✓
```

### Case 2: FEC Fails, ARQ Needed

```text
T=0:        Packet N lost
T=10ms:     Parity arrives, FEC fails
T=15ms:     Receiver sends ACK with bitmap
T=15+RTT/2: Sender receives ACK
T=15+RTT:   Retransmit arrives

Total: ~15ms + RTT ✓ (under 50ms + RTT)
```

### Case 3: ACK Lost, Retry

```text
T=0:        Packet N lost
T=15ms:     First ACK sent (lost)
T=35ms:     Second ACK sent (20ms interval)
T=35+RTT:   Retransmit arrives

Total: ~35ms + RTT ✓ (still under 50ms + RTT)
```

---

## File Structure

```text
src/reliable/
├── mod.rs              # Module exports
├── architecture.md     # This file
├── protocol.rs         # AckPacket, PacketType extension
├── stats.rs            # NetworkStats
├── arq.rs              # SenderArq, ReceiverArq
├── adaptive.rs         # AdaptiveFec tuner
├── bbr.rs              # BBR congestion control
├── encoder.rs          # ReliableEncoder
├── decoder.rs          # ReliableDecoder
├── async_encoder.rs    # AsyncReliableEncoder
├── async_decoder.rs    # AsyncReliableDecoder
└── tests.rs            # Unit tests
```

---

## Implementation Order

| Phase | Task | Dependencies |
|-------|------|--------------|
| 1 | `protocol.rs` - ACK packet format | None |
| 2 | `stats.rs` - NetworkStats | None |
| 3 | `arq.rs` - ARQ state machines | protocol.rs |
| 4 | `adaptive.rs` - Adaptive FEC | stats.rs |
| 5 | `bbr.rs` - BBR congestion control | stats.rs |
| 6 | `encoder.rs` - ReliableEncoder | arq, adaptive, bbr |
| 7 | `decoder.rs` - ReliableDecoder | arq, stats |
| 8 | `async_*.rs` - Async variants | encoder, decoder |
| 9 | `tests.rs` - Unit tests | All |
| 10 | Integration + examples | All |

---

## References

- [BBR Congestion Control](https://research.google/pubs/pub45646/)
- [nyxpsi](../nyxpsi/) - NetworkStats and adaptive redundancy concepts
- [RFC 6298](https://tools.ietf.org/html/rfc6298) - Computing TCP's Retransmission Timer
