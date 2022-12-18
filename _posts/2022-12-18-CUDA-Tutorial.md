출처: [CUDA Tutorial (tistory.com)](https://catcountryroad.tistory.com/22)

**[프로그래밍/CUDA](https://catcountryroad.tistory.com/category/프로그래밍/CUDA)**

### CUDA Tutorial

바람냥냥 2018. 6. 12. 01:39

대표적인 병렬처리 프로그래밍 기법



\- CPU - 복잡한 연산, 단일 성능이 높음 (Clock Speed)

\- GPU - 단순한 연산, 단일 성능이 낮음, ALU 동시에 구동 가능, 프로그래밍에 제한이 있음



A는 한 번에 1개의 공을 옮길 수 있고 1초에 한번 작업을 할 수 있다. (Latency : 1, Throughput : 1)

B는 한 번에 4개의 공을 옮길 수 있고 2초에 한번 작업을 할 수 있다. (Latency : 2, Throughput : 2)

(버스와 스포츠가 라고 생각할 수도 있다.)



전력 효율을 높이기 위해서 CPU 는 점점 latency 를 줄여가고 GPU 는 throughput 을 늘려간다. 

![img](https://t1.daumcdn.net/cfile/tistory/99A829475B1EA35E1B)


**배열 정보, index는 grid size 와 block size로 정의된다.**

- **grid size는 block 수, shape로 결정**
- **block size는 thread 수, shape로 결정**
- **Grid, Block은 1~3차원이 될수 있다.**

- - gridDim.{x,y,z}      - The dimensions of the grid
  - blockDim.{x,y,z}    - The deminsions of the block
  - blockIdx.{x,y,z}     - The index of the current block within the grid
  - threadIdx.{x,y,z}    - The index of the current thread within the block

-  Block 2차원, Thread 3차원으로 구성

| 1차원 배열일 때, 스레드 구성 | BlockIndex | ThreadIndex                                                  |
| ---------------------------- | ---------- | ------------------------------------------------------------ |
| 1차원                        | blockIdx.x | blockIdx.x * blockDim.x + threadIdx.x                        |
| 2차원                        | blockIdx.x | blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x |
| 3차원                        | blockIdx.x | blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockD |

- Block 2차원, Trhead 3차원 구성

| 차원 배열일 때,스레드 구성 | BlockIndex                          | ThreadIndex                                                  |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| 1차원                      | blockIdx.y * gridDim.x + blockIdx.x | BlockIndex * blockDim.x + threadIdx.x                        |
| 2차원                      | blockIdx.y * gridDim.x + blockIdx.x | BlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x |
| 3차원                      | blockIdx.y * gridDim.x + blockIdx.x | BlockIndex * blockDim.z * blockDim.y * blockDIm.x + threadIdx.z * blockDim.y * blockDim.z + threadIdx.y * blockDim.x + threadIdx.x |

![img](https://t1.daumcdn.net/cfile/tistory/99CD09445B1E9E3A23)





![img](https://t1.daumcdn.net/cfile/tistory/995FCB4C5B1EA5C10A)







![img](https://t1.daumcdn.net/cfile/tistory/99F0AB4B5B1EA87608)



\- 사이즌 어떻게 ? 최적화는 ? 

\- How many threads are active at one time!!

\- Total Threads = Grid 수 X 각 Grid의 Block 수 X 각 Block의 Thread수

CUDA 구조체중 dim3로 grid, block수를 정의한 다음의 kernel의 경우

dim3 dimGrid(5,2,1);

dim3 dimBlock(4,3,6);

gridDim.x = 5 -----------> blockIdx.x = 1.........5

gridDim.y = 2 -----------> blockIdx.y = 1.........2

gridDim.x = 1 -----------> blockIdx.z = 1.........1



blockDim.x = 4, ----------> threadIdx.x = 1....4

blockDim.x = 3, ----------> threadIdx.y = 1....3

blockDim.x = 6, ----------> threadIdx.z = 1....6



blockDim.x = 4 , 



**5\*2\*1\*4\*3\*6 = 720**



![img](https://www.3dgep.com/cgi-bin/mathtex.cgi?blockID=(blockIdx.y*gridDim.x)+blockIdx.x)

![img](https://www.3dgep.com/cgi-bin/mathtex.cgi?threadID=(threadIdx.y*blockDim.x)+threadIdx.x)





[Ref]

[Introduction to Cuda](http://www.karimson.com/posts/introduction-to-cuda/)

[GPU Programming( 왜 GPU를 사용하는가 )](http://hiuaa.tistory.com/66)

[3D Game Engine Programming](https://www.3dgep.com/cuda-thread-execution-model/)

[CUDA](https://m.blog.naver.com/PostView.nhn?blogId=sysganda&logNo=30124649583&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F)

[CUDA Processor에 대한 이해](http://haanjack.github.io/cuda/2016/03/31/cuda-processor.html#CUDA Processor Architecture)