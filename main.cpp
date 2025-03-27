//This main file is just to test out the written functions. Ensure the library works as intended
#include <iostream>
#include "linalg.h"

int main()
{
    LinAlg linalg;
    
    int A[4] = {1, 2, 3, 4};
    int B[4] = {5, 6, 7, 8};
    double* result = linalg.add(A, B, 2, 2);  // 2x2 matrix
    std::cout << "Matrix Addition Result:\n";
    for (int i = 0; i < 4; i++)
    {
        std::cout << result[i] << " ";
        if ((i + 1) % 2 == 0) std::cout << std::endl;
    }
	double dt=linalg.dot(A,B,2,2);
	std::cout<<"Dot product = "<<dt<<std::endl;
    return 0;
}