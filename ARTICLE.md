# I Started Writing a Neural Network in C in 1989. Here's What It Looks Like in 2026.

*On building software that outlasts its original hardware, operating system, and programming paradigm.*

---

In 1989 I sat down with a textbook on artificial neural networks and started typing C code. There was no internet to look things up on. There was no Python, no TensorFlow, no GPU compute. There was certainly no Stack Overflow. Just a textbook, a compiler, and the question: *can I make a machine learn?*

That code is still running today.

---

## The Original Machine

The first version of what I now call **nnsd** — the Neural Network Server Daemon — ran as a Novell NetWare Loadable Module (NLM) on a Novell 386 server.

If you weren't programming in 1989, it is hard to convey how significant "386" and "32-bit" were. Most desktop machines were still 16-bit. The 286 had run Windows under a cumbersome segmented memory model; the 386 opened up a full 4 GB flat address space. There were other 32-bit flat-memory operating systems, but they were expensive. Running as an NLM meant running in **ring 0** — the kernel privilege level, no memory protection, no virtual address space boundaries, full access to the hardware and nothing to slow you down. By today's standards that sounds terrifying. At the time it felt like having the keys to the city. Since I knew nothing was going to be running on these machines except my code I wasn't worried about multi-user protections.

Novell NetWare was the dominant network operating system of the era, and it had a serious networking stack. NLMs communicated over **IPX** (packet-based, connectionless, roughly analogous to UDP) and **SPX** (stream-based, connection-oriented, roughly analogous to TCP). The neural network ran as a server process; clients sent it training requests over IPX/SPX and it processed them cooperatively — no preemptive multithreading. The nnsd NLM could handle any number of simultaneous networks training or running as it was fully multi-threaded with mutex protections for common data structures.

The training algorithm was backpropagation with SGD, implemented exactly as described in the textbook. Weights were floating-point numbers. The activation function was sigmoid. That was it. It worked and it was fast for its day. I built a cluster of a dozen 386 machines running in a rack in a spare bedroom and set about making my fortune.

Now I needed a problem to solve.  More specifically, I needed data.  At the time I was consulting for a trading house, and they had interest in what I was describing and offered me forex data.  I knew nothing about forex, but they provided 20 years of daily OHLC for 20 currency pairs (this was pre-Euro, so there were lots of currencies — DEM, FRF, ITL, and a dozen others were still live pairs). Forex doesn't have a centralised market, so there is no real "open" and "close". I used 6pm Eastern as a boundary for a "day". I experimented with different types of feature creation from the data using many types of data normalisation — anything I could find that was published. Using the cluster of 12 machines I trained various networks with various hyperparameter settings on various data sets. Training a single currency pair typically took about 30 days of 100% CPU processing on one machine, so the cluster was running flat out 24 hours a day, 7 days a week. I finally found a profitable combination of settings and data and could train each currency pair to a high percentage of correct predictions, maintaining that accuracy on OOS test sets. Success!  I sold that system back to the firm that provided the data.  A win-win.

---

## Three Decades of Iteration

**1994 — Linux and POSIX**

When Linux emerged as a serious operating system and TCP/IP displaced Novell's proprietary stack, I ported nnsd. The fundamental algorithm didn't change — the mathematics of backpropagation is the same on every platform — but everything around it did. IPX/SPX became TCP/IP sockets. The NLM runtime was replaced by Linux and the pthreads model. The cooperative scheduler became a proper preemptive threading model using POSIX threads and POSIX mutexes. Early stability features — logging and graceful shutdown — were added at this point as well.

**2010 — Increasing Performance**

Another rewrite and I found I could increase performance significantly on newer hardware. I also looked for new problems to solve. This revision completed the operability work: daemon mode, mutex-protected data structures, signal handling, and multiple concurrent networks each in their own thread. The kind of engineering that makes software something you can actually operate rather than just demonstrate.

**2025–2026 — The current version**

The most recent changes are the most substantial since the 1994 port. This is the version I want to describe in detail.

---

## What nnsd Is Today

nnsd is a **neural network training and inference server written in pure C**, compiled with full optimisation (`-O3 -march=native`) for AVX2 auto-vectorisation on modern hardware. It exposes a **REST API** and streams live training progress back to clients over an asynchronous TCP callback channel.

It has no Python. No dynamic languages anywhere in the stack. It links against three C libraries: pthreads, cJSON (for wire serialisation), and libcsv (for reading training data). That's the entire dependency tree.

### The Architecture

The server manages a linked list of **network threads**. When a client sends a train, run, or test request, nnsd allocates a `ThreadStruct`, spawns a background thread, and immediately returns an acknowledgement. The actual work happens asynchronously. Progress updates — passes completed, mean squared error, direction accuracy, elapsed time — are sent back to the client as JSON packets over a TCP connection the client opens and waits on.

This design was chosen in the early 2000s for a reason that still holds: training a non-trivial network takes time. No client should block for that long. The async callback model means a client can display a live progress bar, set a timeout, or cancel training mid-run and the server handles it cleanly.

```
Client                              nnsd server
  |                                     |
  |--- POST /api/networks/16/retrain -->|  (returns immediately)
  |                                     |  [spawns training thread]
  |<-- STATS (passes=5000, MSE=0.24) --|  (every 4 seconds)
  |<-- STATS (passes=12000, MSE=0.18)--|
  |<-- STATS (passes=28000, MSE=0.12)--|
  |--- POST /api/networks/16/cancel  -->|  (if timeout reached)
  |<-- NETFINISHED (weights saved)   --|
```

### Mini-Batch Training with a Thread Pool

The most significant performance addition in the current version is a **persistent worker thread pool** for mini-batch gradient accumulation.

On a 28-core machine, 28 worker threads are created at training start and sleep on semaphores between batches. Each training pass is divided into mini-batches; each mini-batch is dispatched to the worker pool. Workers compute forward and backward passes independently on their slice of data, accumulating gradients into per-worker local arrays. The main training thread then waits for all workers to finish, sums the accumulated gradients, and applies the weight update.

The result: the inner loop — which is just dense floating-point multiply-adds over weight matrices — is embarrassingly parallel and vectorises cleanly with AVX2. On the 60,000-sample MNIST dataset, a full training epoch takes about 2 seconds wall-clock on a modern desktop.

### Activation Functions

Four activation functions are supported, selectable per network at training time:

| Function   | Formula           | Use case                                         |
| ---------- | ----------------- | ------------------------------------------------ |
| Sigmoid    | 1 / (1 + e^−z)    | Legacy; binary classification                    |
| Tanh       | tanh(z)           | Preferred for classification; output in [−1, +1] |
| ReLU       | max(0, z)         | Deep networks; avoids vanishing gradient         |
| Leaky ReLU | z > 0 ? z : 0.01z | ReLU without dead neurons                        |

For binary classification problems I use tanh: the output naturally lives in (−1, +1), which means the training targets {−1, +1} don't need normalisation. The signed output value is directly interpretable as a confidence score — how far the network is from the decision boundary.

### Two Optimisers: SGD and Adam

The original 1989 implementation used stochastic gradient descent with momentum:

```
Δw = α · δ · v + γ · Δw_prev
```

This still works. For small, well-conditioned problems it is hard to beat. But for the kind of high-dimensional, noisy, small-dataset problems I'm working with now — a few hundred training samples, a dozen input features — Adam converges significantly faster.

The current version implements **Adam (Adaptive Moment Estimation)** natively in C. Per-weight first and second moment arrays are allocated alongside the weight matrix. The update rule:

```
m_t = β1·m_{t-1} + (1−β1)·g_t          # first moment (gradient mean)
v_t = β2·v_{t-1} + (1−β2)·g_t²         # second moment (gradient variance)
m̂_t = m_t / (1−β1^t)                   # bias-corrected
v̂_t = v_t / (1−β2^t)                   # bias-corrected
w   = w − α · m̂_t / (√v̂_t + ε)
```

In practice, switching from SGD to Adam on a 250-sample financial time-series dataset cut training time from 30 minutes to under 10 minutes for equivalent convergence. For a server that has been doing SGD since 1989, that is a meaningful upgrade.

### Server-Side Training Timeout

This one sounds simple but it matters operationally. In earlier versions, training stopped when the network converged to the specified accuracy, or when the client sent a CANCEL packet. If the client disconnected unexpectedly — session crash, network drop, user hit Ctrl-C — the server would continue training the network, potentially indefinitely if the specified convergence accuracy (% correct) could not be achieved. Clients could always reconnect to the server, but if they abandoned the network it could run forever burning CPU because no CANCEL ever arrived. The weight matrix is periodically saved, so no training is ever "lost", but orphaned sessions were a problem.

The `trainingTimeout` field (in seconds, 0 = unlimited) is now embedded in every train request. The server checks elapsed time at the top of each training epoch. When the timeout fires, it saves weights, sends a NETFINISHED callback, and cleans up. No orphaned training threads, no silent CPU burn.

### The REST API

The wire protocol has migrated from the original IPX/SPX custom packets through raw sockets and now to a proper REST API over HTTP on port 8080. Every operation is a POST or GET with a JSON body. The packet structure — 35 fields covering network configuration, file paths, optimiser parameters, training statistics, and control flags — is serialised and deserialised using cJSON.

```
POST /api/networks/{id}/retrain    # train from scratch (random weights)
POST /api/networks/{id}/train      # resume from existing weight matrix
POST /api/networks/{id}/run        # inference pass
POST /api/networks/{id}/test       # evaluate accuracy on labelled data
POST /api/networks/{id}/cancel     # stop training cleanly
POST /api/networks/{id}/nudge      # perturb weights to escape local minimum
GET  /api/coef                     # system health: load, free memory, free swap
```

Client libraries exist in C++ (a full async client with its own callback listener and TUI) and in R (a REST wrapper used for financial modelling). Any language that can speak HTTP and JSON can talk to nnsd.

### Data Format

Input data is CSV. The server reads training, test, and inference files directly from a shared mount point — typically `/opt/neural/data`. Ground truth can appear as the first column or the last. For classification problems, a single integer label column can be expanded to one-hot encoding automatically. Output files follow the same convention: predicted class or raw output values, with an optional label-first ordering for easy comparison with ground truth.

Weight matrices are serialised to a binary `.mtx` file between sessions. If a matrix file exists when a `train` request arrives, training resumes from those weights. The `retrain` command deletes any existing matrix file first — forcing random initialisation, useful when the network architecture changes.

---

## What Has Not Changed in 37 Years

The core algorithm. Backpropagation computes gradients by the chain rule, layer by layer, from output back to input. This is unchanged from the textbook I was reading in 1989. The mathematics is the same mathematics.

What has changed is everything else: the hardware it runs on, the operating system, the networking protocol, the API surface, the optimiser, the parallelism model. The soul of the thing — a network of weighted connections learning from examples by gradient descent — is identical to what I typed in on that 386.

There is something clarifying about having maintained a piece of software across that many generations of technology. You develop a very clear sense of what is essential (the algorithm, the weight representation, the training loop) and what is accidental complexity (the wire format, the concurrency model, the API style). The accidental complexity changes with the era. The essential parts don't.

---

## What's Next

A few things remain on the list:

**Dropout** — randomly zeroing activations during training is one of the most effective regularisation techniques for small datasets, where overfitting is the main enemy. It fits naturally into the existing mini-batch worker model.

**Ensemble support** — the server currently manages one active operation per network ID. Clients can ensemble multiple networks. Supporting ensembles on the server side via one API call would be better. Supporting multiple weight files per ID (seeded from different random initialisations) and averaging their inference outputs would significantly improve prediction stability without any changes to the training algorithm.

**Mini-batch size as a parameter** — currently the batch size is fixed to the number of worker threads. Making it configurable would allow tuning the exploration/exploitation tradeoff during training.

---

## On Building Things That Last

People sometimes ask whether it makes sense to write neural network code from scratch in C in an era when PyTorch exists. That is the wrong question.

The reason nnsd exists is not that PyTorch is inadequate. PyTorch is extraordinary. The reason nnsd exists is that I wanted to understand what was happening inside the black box well enough to change it — to add a new activation function, to swap in a different optimiser, to measure exactly what the inner loop costs on real hardware. When you write the code yourself you have that understanding permanently. When you use a framework you borrow it.

The 1989 version of this code taught me backpropagation more thoroughly than any course ever could. The 2025 version — where I had to implement Adam correctly, thread-safe, inside a live server that handles multiple concurrent networks while streaming async status updates to clients — taught me things about numerical stability and parallel gradient accumulation that I would not have learned any other way.

Thirty-seven years of the same program. I recommend it.

---

*nnsd is a private project. The R client and financial modelling application built on top of it are available at [github.com/costab06/forex_stat_arb](https://github.com/costab06/forex_stat_arb).*
