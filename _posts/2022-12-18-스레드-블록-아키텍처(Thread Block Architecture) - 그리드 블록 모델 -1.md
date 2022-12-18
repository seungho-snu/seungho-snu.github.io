출처: [스레드 블록 아키텍처(Thread Block Architecture) - 그리드 블록 모델 -1 :: Todo's Diary (tistory.com)](https://tododiary.tistory.com/51)



## [스레드 블록 아키텍처(Thread Block Architecture) - 그리드 블록 모델 -1](https://tododiary.tistory.com/51)

\2019. 2. 8. 17:02

**02. 그리드 블록 모델**

**
**

CUDA에서 단순하게 스레드 수만 개를 설정했다고 해서 모든 코어가 효율적으로 동작하는 것은 아니다. 



**스레드의 계층 구조 : 스레드 < 블록 < 그리드**

**
**



![img](https://t1.daumcdn.net/cfile/tistory/99C8C04F5C5D22AB3C)







------

**
**

**
**

**2.1 CUDA 블록과 1차원 스레드 생성
**

**
**

**-CUDA의 블록**은 **스레드가 모인 집합**이다. 

-스레드가 여러 개 모여 하나의 블록을 이루게 된다.

-하나의 블록은 1개에서부터 **최대 512개** 까지 스레드를 가질 수 있다.







![img](https://t1.daumcdn.net/cfile/tistory/9975644A5C5D23BC3C)





-블록 안에 있는 스레드는 고유 아이디를 가지게 된다. 

-블록이 스레드를 512개까지 가지고 있을 수 있기 때문에 0번부터 511번 까지 고유한 인덱스 번호를 지정받게 된다.

-블록 안에 스레드를 배치하는 방법은 1차원, 2차원, 3차원으로 지정할 수 있다.







**| 1차원으로 배치한 스레드 블록**





![img](https://t1.daumcdn.net/cfile/tistory/995F444C5C5D24E808)

(블록 안에 6개의 스레드를 생성하여 1차원으로 배치했을 경우를 나타내고 있다.)







스레드 인덱스는 '**threadIdx**' 로 이름이 지정된 변수를 사용한다.



**스레드 인덱스의 용도** : 전체 작업을 분할하여 각 스레드가 해야 할 작업을 분류하는 용도



설정된 스레드의 차원에 따라 threadIdx 는 x, y, z 에 대한 값을 가지게 된다. 

위의 그림과 같이 1차원으로 배치된 스레드의 인덱스는 다음과 같이 구성된다.



스레드 0 : threadIdx.x = 0

스레드 1 : threadIdx.x = 1

스레드 2 : threadIdx.x = 2

스레드 3 : threadIdx.x = 3

스레드 4 : threadIdx.x = 4

스레드 5 : threadIdx.x = 5





**| 문법**



스레드 블록을 생성하여 배치하는 방법은 커널 함수의 '**<<< >>>**' 안에 설정하게 된다.





__global__ void kernel<<<1, 6>>>(int a, int b, int c);







------





CUDA 스레드 모델에서 **SP**는 **4개**의 스레드를 동시에 실행할 수 있고, 하나의 블록은 하나의 SM과 대응하여 동작하게 된다.





**| 스레드 , 블록의 디바이스 대응**



![img](https://t1.daumcdn.net/cfile/tistory/99EEBB435C5D28750B)





1개의 블록에서 512개의 스레드를 생성하면 다음과 같은 코드가 된다.





__global__ void kernel<<<1, 512>>>(int a, int b, int c);





스레드 0 : treadIdx.x = 0

스레드 1 : treadIdx.x = 1

스레드 2 : treadIdx.x = 2

스레드 3 : treadIdx.x = 3

.....

스레드 510 : treadIdx.x = 510

스레드 511 : treadIdx.x = 511







1개의 블록에서 스레드를 생성시켜 프로그램을 실행시키면 **1개**의 SM에서만 동작하여 GPU의 성능을 전부 발휘하지 못하고 비효율적이 된다.





![img](https://t1.daumcdn.net/cfile/tistory/99911D335C5D2A631A)







다음과 같이 수정하면 **8개**의 SM을 동작시킬 수 있다.





__global__ void kernel<<<8, 64>>>(int a, int b, int c);





블록을 1차원으로 여러개 생성하면 다음과 같은 인덱스를 가지게 된다.



블록 0 : blockIdx.x = 0

블록 1 : blockIdx.x = 1

블록 2 : blockIdx.x = 2

.......

블록 6 : blockIdx.x = 6

블록 7 : blockIdx.x = 7





**| 8개 블록 64개 스레드로 구성된 스레드 블록 인덱스**

**
**

**
**

![img](https://t1.daumcdn.net/cfile/tistory/99C9603C5C5D2CA61C)





블록마다 스레드 인덱스는 0부터 63번까지 생성된다.





블록과 스레드가 함께 구성된 경우 개별적인 스레드를 분류하려면 다음과 같은 수식을 사용해야 한다.



//blockDim.x 는 1차원 블록의 스레드 개수(크기)

int tid = blockIdx.x * blockDim.x + treadIdx.x;





SP는 4개의 스레드를 실행하고 SM은 32개의 스레드를 동작시키기 때문에 64개의 스레드로 구성된 블록은 32개를 먼저 실행시키고 다음 32개의 스레드는 나중에 실행하게 된다.







------







**2.2 두 벡터의 합 계산**





3000만개 정도의 스레드를 생성해보자. 아래의 예제는 1차원 스레드 블록 생성을 이용하여 두 벡터의 합을 계산하는 프로그램을 작성하는 것이다.



배열의 크기는 1차원에서 CUDA가 생성할 수 있는 스레드의 최댓값으로 한다. 1차원 배열이 가질 수 있는 최대 스레드 개수는 512개개 이고 그리드가 가질 수 있는 블록의 수는 65,535개 이므로 512 X 65,535 = 33,553,920 개의 스레드를 생성할 수 있다.



입력 데이터를 위한 정수형 배열 InputA, InputB에 대한 요소 33,553,920개를 할당하고, 출력 데이터를 위한 정수형 배열 Result도 같은 크기로 할당한다.



//호스트 메모리 할당

InputA = (int *)malloc(512*65535*sizeof(int));

InputB = (int *)malloc(512*65535*sizeof(int));

InputC = (int *)malloc(512*65535*sizeof(int));







입력데이터를 위한 정수형 배열 InputA 와 InputB에 0부터 33,553,919까지의 값을 입력하여 두 배열의 값을 더하여 결과를 Result 배열에 넣는다.





**| 두 배열의 합 계산**

**
**

**
**

![img](https://t1.daumcdn.net/cfile/tistory/99DF634A5C5D312816)





<위의 과정에 대한 예제>



\#include <cuda_runtime.h>

\#include <stdio.h>



__global__ void VectorAdd( int *a, int *b, int *c, int size)

{

int tid = blockIdx.x * blockDim.x + threadIdx.x;



c[tid] = a[tid] + b[tid];

}



int main()

{

const int size = 512 *65535;

const int BufferSize = size*sizeof(int);



int* InputA;

int* InputB;

int* Result;



//호스트 메모리 할당

InputA = (int*)malloc(BufferSize);

InputB = (int*)malloc(BufferSize);

Result = (int*)malloc(BufferSize);



int i = 0;



//데이터 입력

for(int i = 0; i < size; i++)

{

InputA[i] = i;

InputB[i] = i;

Result[i] = 0;

}



int* dev_A;

int* dev_B;

int* dev_R;



//디바이스 메모리 할당

cudaMalloc((void**)&dev_A, size*sizeof(int));

cudaMalloc((void**)&dev_B, size*sizeof(int));

cudaMalloc((void**)&dev_R, size*sizeof(int));



//호스트 디바이스 입력 데이터 전송

cudaMemcpy(dev_A, InputA, size*sizeof(int), cudaMemcpyHostToDevice);

cudaMemcpy(dev_B, InputB, size*sizeof(int), cudaMemcpyHostToDevice);



//33,553,920개의 스레드를 생성하여 덧셈 계산

VectorAdd<<<65535, 512>>>(dev_A, dev_B, dev_R, size);



//디바이스 호스트 출력 데이터 전송

cudaMemcpy(Result, dev_R, size*sizeof(int), cudaMemcpyDeviceToHost);



//결과 출력

for( i = 0; i < 5; i++)

{

printf("Result[%d] : %d\n", i , Result[i]);

}

printf("......\n");

for( i = size -5; i< size; i++)

{

printf("Result[%d] : %d\n", i, Result[i]);

}



//디바이스 메모리 해제

cudaFree(dev_A);

cudaFree(dev_B);

cudaFree(dev_R);



//호스트 메모리 해제

free(InputA);

free(InputB);

free(Result);



return 0;

}



**| 커널 함수**



//33,553,920개의 스레드를 생성하여 덧셈 계산

VectorAdd<<<65535, 512>>>(dev_A, dev_B, dev_R, size);





커널함수를 1차원에서 최대 실행할 수 있는 스레드의 개수로 생성하고 하나의 스레드는 하나의 배열 요소를 계산하도록 구현하였다.









__global__ void VectorAdd( int *a, int *b, int *c, int size)

{

int tid = blockIdx.x * blockDim.c + treadIdx.x;



c[tid] = a[tid] + b[tid];

}









33,553,920개의 배열을 하나도 빠짐없이 스레드로 계산하기 위해 고유의 스레드 인덱스를 계산하여 배열 인덱스와 연결한다. 다음의 계산식은 스레드 번호를 0부터 33,553,929까지 정렬해준다.





int tid = blockIdx.x * blockDim.x + treadIdx.x;





**| 두 벡터의 값 계산 결과**

**
**

![img](https://t1.daumcdn.net/cfile/tistory/99BB6B375C5D38CC21)