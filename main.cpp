//This main file is just to test out the written functions. Ensure the library works as intended
#include <iostream>
#include "linalg.h"

int main()
{
    LinAlg linalg;
    
    int A[2][2] = {{1, 2}, {3, 4}};
    int B[2][2] = {{5, 6}, {7, 8}};
	int C[4] = {1, 2, 3, 4};
    int D[4] = {5, 6, 7, 8};
    double (*result)[2] = linalg.add(A, B, 2);  // 2x2 matrix
    std::cout << "Matrix Addition Result:\n";
    for (int i = 0; i < 2; i++)
    {
		for (int j=0; j<2; j++)
		{
			std::cout << result[i][j] << " ";
		}
		std::cout << std::endl;
    }
	double dt=linalg.dot(C,D,4);
	std::cout<<"vector dot product"<<std::endl<<dt<<std::endl;
	
	double (*idt)[5]=linalg.identity<5>();
	
	std::cout << "Identity Matrix Result:\n";
    for (int i = 0; i < 5; i++)
    {
		for (int j=0; j<5; j++)
		{
			std::cout << idt[i][j] << " ";
		}
		std::cout << std::endl;
    }
	
	double (*matm)[2]= linalg.multiply(A,B,2,2);
	std::cout << "Matrix Multiplication Result:\n";
    for (int i = 0; i < 2; i++)
    {
		for (int j=0; j<2; j++)
		{
			std::cout << matm[i][j] << " ";
		}
		std::cout << std::endl;
    }
	
	double (*cp)[2]= linalg.copy(A,2);
	std::cout << "Matrix Copy Result:\n";
    for (int i = 0; i < 2; i++)
    {
		for (int j=0; j<2; j++)
		{
			std::cout << cp[i][j] << " ";
		}
		std::cout << std::endl;
    }
	
	double (*pows)[2]= linalg.power(A,0);
	std::cout << "Matrix Power Result:\n";
    for (int i = 0; i < 2; i++)
    {
		for (int j=0; j<2; j++)
		{
			std::cout << pows[i][j] << " ";
		}
		std::cout << std::endl;
    }
	
    return 0;
}