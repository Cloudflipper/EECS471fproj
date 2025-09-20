# EECS 471 W25 FINAL REPORT
### Kangqi Zhang, Mingrui Li
In this project, we adopt the following optimizations:
1. Multi-dimensional Parallelization 
2. Constant Memory Optimization
3. Loop Unrolling 
4. Shared Memory Tiling
5. Shared Memory Layout Optimization

The following chart shows the improvement of the performance
|        | Req | Starter |  O1   |  O12  |  O13  | O134  | O1~4  | O1~5  |  T  |
|:------:|:---:|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| Layer1 | N/A |  3030   | 36.1  | 32.0  | 31.5  | 33.9  | 22.4  | 23.6  | 14  |
| Layer2 | N/A |  11970  | 137.6 | 185.3 | 121.9 | 144.8 | 115.1 | 90.7  | 20  |
| Total  | 200 |  15000  | 173.7 | 217.3 | 153.4 | 188.7 | 137.5 | 114.3 | 34  |
All number are in unit ms. Req means requirement, starter means starter code, O means Optimization. O12 means optimization 1 and 2, O1~4means optimization 1~4, and so on. T means built-in Pytorch.

## 1.Multi-dimensional Parallelization
### Implementation
The final code restructures parallelism across three GPU execution dimensions to maximize hardware utilization, diverging from the original single-dimensional thread mapping.
#### Original Single-Dimensional Parallelism

> Code Deleted for release

In original code, onlybatch dimension (B) are parallelized. There are 6 nested loops inside each thread. $(M×H_{out}×W_{out}×C×K×K)$. This is a very severe underutilization of GPU threads and incorporates no spatial locality optimization.
#### Optimized 3D Grid + 2D Block Parallelism
The final code decomposes computation across three GPU execution dimensions:

> Code Deleted for release

1. BlockIdx.z: Processes different input images in batch 
1. BlockIdx.y: Computes separate output feature maps
1. BlockIdx.x: Handles spatial tiling of output dimensions

Each thread block processes a TILE_SIZE × TILE_SIZE output region for one specific (batch, output channel) pair. Threads within a block use 2D indexing to cover spatial coordinates, while grid dimensions explicitly separate batch, channel, and spatial parallelism

After adjusting parameters, the ```TILE_SIZE``` is determined to be 16.

#### Access Pattern


> Code Deleted for release


This incorporates massive parallelism, where there are approximately $gridDim.x × gridDim.y × gridDim.z × blockDim.x × blockDim.y= H_{out}×W_{out}×M×B$
Threads in parallel

This enables Memory Access Coalescing, because threads in warp access consecutive output positions, because ```threadIdx.x``` corresponds to the output column index within tile, so it chieves perfect 128-byte coalesced memory writes to y.

### Consequence
|         | Req/s | Starter Code/s | Optim1/s |
|:-------:|:-----:|:--------------:|:--------:|
| Layer 1 |  N/A  |      3.03      |  0.0361  |
| Layer 2 |  N/A  |     11.97      |  0.1376  |
|  Total  | 0.20  |     15.00      |  0.1737  |

## 2. Constant Memory Optimization
### Implementation
The final code leverages CUDA's constant memory to store convolution filter weights, optimizing access patterns during the sliding window computation.
#### Constant Memory Declaration & Initialization

> Code Deleted for release

The filter weights are declared in constant memory using the constant qualifier, which places them in a cache. During kernel launch, weights are copied from host memory to this persistent constant memory region via ```cudaMemcpyToSymbol```, eliminating repeated transfers.
#### Kernel Access Pattern

> Code Deleted for release

All filter weight accesses now target the constant memory array rather than a global memory pointer.
### Consequence

|         | Req/s | Optim1/s | Optim1+2/s |
|:-------:|:-----:|:--------:|:----------:|
| Layer 1 |  N/A  |  0.0361  |   0.0320   |
| Layer 2 |  N/A  |  0.1376  |   0.1853   |
|  Total  | 0.20  |  0.1737  |   0.2173   |

**There is not a better performnce, so we abort this change.**
## 3.Loop Unrolling
### Implementation
The final code replaces nested filter loops with explicit arithmetic operations to eliminate control flow overhead and maximize hardware utilization. This optimization targets the 7×7 convolution kernel’s innermost computation.
#### Original Loop Structure

> Code Deleted for release

The baseline implementation uses two nested loops (7×7 iterations) to process filter elements. Each iteration requires loop counter management (14 condition checks for loop boundaries) and generates fragmented memory access patterns.
#### Manual Loop Unrolling Implementation

> Code Deleted for release

The optimized version explicitly writes all 49 filter position calculations, eliminating loop control logic. This transforms the computation into a sequence of independent fused multiply-add (FMA) operations. Threads now execute 49 direct arithmetic operations without branch divergence or counter updates.
### Consequence
|         | Req/s | Optim1/s | Optim1+3/s |
|:-------:|:-----:|:--------:|:----------:|
| Layer 1 |  N/A  |  0.0361  |   0.0315   |
| Layer 2 |  N/A  |  0.1376  |   0.1219   |
|  Total  | 0.20  |  0.1737  |   0.1534   |
### Remark
In fact, we also attempted unrolling for the channel layer. However, tests showed that unrolling the channel layer did not contribute to improving the computational speed. Therefore, we abandoned this approach.






## 4.Shared Memory Tiling
### Implementation
The final code implements shared memory tiling to dramatically reduce global memory accesses in convolution operations. This optimization fundamentally changes how input data is accessed and reused across threads.
####  Shared Memory Allocation & Configuration

> Code Deleted for release

The implementation begins by declaring a shared memory buffer. A three-dimensional indexing scheme is established through the Ns_idx(c, i, j), which maps input channels (indexed by c), rows (i), and columns (j) into a contiguous shared memory space. The halo tile size is computed as HALO_TILE = TILE_SIZE + K - 1, where TILE_SIZE defines the output region processed per thread block, and K is the filter size. This configuration accounts for the filter’s overlap between adjacent tiles.

Also, it is worth note that the shared memory per block$$size=C×(HALO\_TILE)^2 ×sizeof(float)=3×22^2×4=5808B$$This memory is much smaller than the 48KB limit per block.
#### Halo Data Loading between threads
This part performs halo data loading:

> Code Deleted for release

Threads collaboratively load input data from global memory into shared memory using a strided access pattern. Each thread block processes a spatial region of size HALO_TILE × HALO_TILE, which includes both the output tile and surrounding halo pixels required for convolution. Threads iterate over the halo region in strides equal to the block dimensions, enabling coalesced global memory accesses. 
For example, a thread with indices (tx, ty) loads data at positions (tx, ty), (tx + blockDim.x, ty), and (tx, ty + blockDim.y) until the entire halo region is covered. Boundary checks ensure out-of-range coordinates are padded with zeros, emulating "valid" convolution semantics.
#### Convolution Computation with Explicit FMA Unrolling

> Code Deleted for release

The 7×7 convolution kernel is manually unrolled into 49 operations. For each output position (row_o, col_o), the computation iterates over input channels and accumulates results using filter weights stored in constant memory. Shared memory accesses leverage preloaded data offsets: the thread’s local indices (tx, ty) correspond to the top-left corner of the filter’s application window. All 49 filter positions are explicitly addressed using fixed offsets. Results are accumulated in a register variable (output) to minimize intermediate memory traffic.

### Consequence
|         | Req/s | Optim1+3/s | Optim1+3+4/s | Optim1~4/s |
|:-------:|:-----:|:----------:|:------------:|:----------:|
| Layer 1 |  N/A  |   0.0315   |    0.0339    |     0.0224        |
| Layer 2 |  N/A  |   0.1219   |    0.1448    |       0.1151     |
|  Total  | 0.20  |   0.1534   |    0.1887    |       0.1375     |
Although adding Optimization 2 and Optimization 4 individually did not significantly improve the performance, combining both Optimization 2 and Optimization 4 led to a significant improvement.


## 5.Shared Memory Layout Optimization
### Implementation
The final code redesigns shared memory organization to enhance data locality and reduce bank conflicts during convolution computations.
#### Original Memory Layout

> Code Deleted for release

The baseline implementation uses a spatial-first layout where channels are stored as separate blocks and each channel's data is arranged in ```HALO_TILE```x```HALO_TILE``` grids.
#### Optimized Memory Layout

> Code Deleted for release

The optimized version introduces channel-contiguous layout and reduces bank conflict, where stride patterns align with GPU memory banks.
### Consequence
|         | Req/s | Optim1~4/s | Optim1~5/s |
|:-------:|:-----:|:----------:|:----------:|
| Layer 1 |  N/A  |   0.0224   |   0.0236   |
| Layer 2 |  N/A  |   0.1151   |   0.0907   |
|  Total  | 0.20  |   0.1375   |   0.1143   |

## Attached: Original Code
> Code Deleted for release
