# RunPod Hardware Selection Guide: AlphaZero Self-Play Optimization

When deploying AlphaZero self-play (`train` mode or `match` mode) on RunPod, raw hardware specifications (like core count or peak TFLOPS) can be highly misleading. 

This guide documents a recent real-world performance analysis comparing two distinct RunPod instances, explaining why a lower-spec instance on paper achieved **nearly double** the throughput of a higher-spec instance. Use these guidelines to make the most cost-effective and high-performance rentals in the future.

---

## 📊 The Paradox: Case Study of Two Pods

| Metric | Pod A (High Performance 🚀) | Pod B (Lower Performance 🐢) |
| :--- | :--- | :--- |
| **Measured Throughput** | **~800 - 855 positions/second** | **~487 positions/second** |
| **GPU** | 1x NVIDIA V100 SXM2 | 1x NVIDIA RTX 3090 |
| **GPU Peak FP32 Compute**| ~15.7 TFLOPS | ~35.6 TFLOPS |
| **GPU Interface / Form** | Enterprise **SXM2 / NVLink** | Consumer **PCIe** |
| **vCPU Count** | **10 vCPUs** | **32 vCPUs** |
| **CPU Architecture** | Intel Xeon E5-2698 v4 (Broadwell) | AMD EPYC 7763 (Zen 3) |
| **CPU Silicon Layout** | **Monolithic Ring-Bus** | **Multi-Chiplet (CCD) / NUMA** |
| **System Memory** | 62 GB | 125 GB |
| **Hourly Cost** | ~$0.20 - $0.30 (V100 SXM2 is cheap!) | ~$0.30 - $0.40 |

### Why did Pod A outperform Pod B by ~75% despite having 1/3 of the CPU cores and half the GPU compute?

---

## 🧠 1. CPU Bottleneck: NUMA/Chiplets vs. Monolithic Cache

### The MCTS Workload Profile
Monte Carlo Tree Search (MCTS) is an intensively **memory-latency-bound** workload. It consists of millions of traversals of heap-allocated, pointer-heavy tree structures. MCTS threads perform continuous dynamic reads/writes to node counts, prior probabilities, and Q-values. It is characterized by heavy **pointer-chasing** and CPU cache utilization, rather than dense arithmetic.

### The AMD EPYC 7763 (Zen 3) Chiplet Penalty
* **Multi-Chiplet Architecture:** Modern AMD EPYC and Ryzen processors consist of multiple Core Complex Dies (CCDs) connected via a central I/O die over an **Infinity Fabric** interconnect.
* **The Latency Trap:** When running 32 concurrent MCTS threads, threads are distributed across multiple CCDs. If a thread needs to access tree nodes or synchronization states residing in memory/cache controlled by a different CCD, it must cross the Infinity Fabric.
* **Cache-Coherency Thrashing:** Even though our C++ engine utilizes a highly efficient **lock-free queue** (`boost::lockfree::queue`), the hardware cache-coherency protocols must constantly synchronize memory states across physically separate chiplets. The inter-CCD memory access latency ruins MCTS generation speed.

### The Intel Xeon Monolithic Advantage
* **Monolithic Ring Bus:** Older or monolithic server CPUs (like the Intel Xeon E5 v4 family) contain all cores on a single physical silicon die, connected via a high-speed unified ring bus.
* **Uniform Cache Latency:** All 10 cores share a large, uniform L3 cache with predictable, ultra-low latency. Because there are no CCD boundaries or complex NUMA hops on a single socket, thread communication and lock-free queue pushes complete in a fraction of the time.
* **The Sweet Spot:** Optuna naturally discovered that on Pod B (AMD EPYC), it had to constrain `parallel_games` to **just 17** (despite having 32 vCPUs) because increasing the thread count further caused massive cache-thrashing and memory bus saturation.

> [!IMPORTANT]
> **CPU Rule of Thumb:** For AlphaZero self-play, **fewer, high-frequency monolithic cores** (or cores isolated to a single CCD / NUMA node) are vastly superior to **many cores spread across multi-chiplet topologies**. 
> * Prioritize high single-core clock speeds (GHz) over high core count.
> * If using AMD EPYC, pin the process to a single CCD/NUMA node using `numactl` or `taskset` if possible.

---

## ⚡ 2. GPU Bottleneck: SXM2 / NVLink vs. PCIe

### The Self-Play Inference Profile
Unlike neural network training (which processes massive, static, contiguous batches), RL self-play operates in a tight, high-frequency loop:
1. MCTS threads request neural network evaluations continuously.
2. The engine batches these requests in a lock-free queue.
3. The inference loop drains the queue, copies the input tensor from Host-to-Device (H2D), runs the forward pass, and copies the policy/value output from Device-to-Host (D2H) to unlock MCTS threads.

### Host-to-Device (H2D) Transfer Latency is King
* **The PCIe Bottleneck:** The RTX 3090 is a consumer card running on a PCIe interface. For frequent, small-to-medium transfers (low latency round-trips), standard PCIe buses introduce significant software and hardware interrupt overhead. The GPU sits idle waiting for the next batch to be dispatched.
* **The SXM2/NVLink Advantage:** Enterprise form-factor GPUs (like V100 SXM2 or A100 SXM4) plug directly into proprietary motherboard sockets that bypass PCIe limitations, offering ultra-high bandwidth and extremely low latency NVLink topologies. The round-trip overhead of dispatching a batch is microscopic.
* **Peak TFLOPS is a Trap:** The RTX 3090's 35.6 TFLOPS are starved because it spends most of its time waiting on CPU memory dispatch and PCIe bus transfer latency. The V100 SXM2's lower 15.7 TFLOPS are highly saturated because data flows back and forth instantly.

> [!TIP]
> **GPU Rule of Thumb:** For self-play RL, choose enterprise **SXM2 / SXM4** form-factors (e.g., V100 SXM2, A100 SXM4) over consumer **PCIe** cards (RTX 3090, RTX 4090). 
> * The V100 SXM2 is frequently available on RunPod at deep discounts and will actively outperform more expensive PCIe setups for RL self-play.

---

## 🔄 3. Interaction of Virtual Loss & Lock-Free Queues

Our C++ engine utilizes a non-blocking **lock-free queue** (`boost::lockfree::queue`) combined with **virtual loss** during MCTS.

* **Non-Blocking Execution:** Because the queue is lock-free, threads never sleep or block on mutex locks when submitting evaluations.
* **Virtual Loss Buffer:** When a thread sends an evaluation request to the queue, it applies a virtual loss penalty to the node. This allows the thread to continue searching other branches in the game tree without waiting for the GPU to respond.
* **Queue Size Independence:** Because threads do not block, the `max_batch_size` acts as a queue/buffer capacity. If the queue is too small, MCTS threads will starve.
* **Why Optuna Chose 2768 Batch Size:** On the AMD EPYC/RTX 3090 pod, Optuna chose `max_batch_size: 2768` and `parallel_games: 17`. Because the CPU-to-GPU transfer was a bottleneck (PCIe), Optuna realized that accumulating a massive queue of requests and evaluating them in large, less-frequent batches minimized the high-frequency PCIe dispatch overhead, maximizing overall throughput.

---

## 📋 RunPod Rental Cheat Sheet

Use this checklist next time you rent a pod on RunPod for AlphaZero self-play:

1. **Prefer SXM2/SXM4 over PCIe:**
   * Look for **V100 SXM2**, **A100 SXM4**, or **A30 (SXM)**.
   * Avoid consumer **RTX 3090/4090** unless they are paired with high-performance monolithic CPUs and represent a significant cost saving.

2. **Verify CPU Architecture:**
   * Choose **Intel Xeon** or **AMD EPYC with high single-core clocks**.
   * If renting an EPYC instance, try to rent one with high RAM speed and fewer, faster cores, rather than 32+ slow cores.

3. **Check Cgroups to Prevent Host CPU Misreporting:**
   * Python's `os.cpu_count()` reports the host's physical cores, but RunPod limits your container via cgroups.
   * Our updated code uses `get_usable_cpu_count()` to dynamically inspect `/sys/fs/cgroup/cpu.max` so Optuna bounds are always 100% accurate to your rented limits!

4. **Tune Bounds Separately:**
   * Keep `parallel_games` close to the actual usable CPU count.
   * Let `max_batch_size` range freely (e.g., 64 to 4096) to act as a buffer, allowing the lock-free queue and virtual loss to absorb hardware latency.
