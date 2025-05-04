//This main file is just to test out the written functions. Ensure the library works as intended
#include "linalg.h"
#include <iomanip>

int main()
{
	LinAlg linalg;
    
    	int A[2][2] = {{1, 2}, {3, 4}};
    	int B[2][2] = {{5, 6}, {7, 8}};
	int C[4] = {1, 2, 3, 4};
    	int D[4] = {5, 6, 7, 8};
	int E[2] = {3,7};

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
	
	double (*pows)[2]= linalg.power(A,5);
	std::cout << "Matrix Power Result:\n";
    	for (int i = 0; i < 2; i++)
    	{
		for (int j=0; j<2; j++)
		{
			std::cout << pows[i][j] << " ";
		}
		std::cout << std::endl;
    	}
	
	double (*otr)[4]= linalg.outer<4>(C, D);
	std::cout << "Vector Outer Product Result:\n";
    	for (int i = 0; i < 4; i++)
    	{
		for (int j=0; j<4; j++)
		{
			std::cout << otr[i][j] << " ";
		}
		std::cout << std::endl;
    	}
	
	double (*tp)[2]= linalg.transpose<2>(A);
	std::cout << "Transpose of a Matrix Result:\n";
    	for (int i = 0; i < 2; i++)
    	{
		for (int j=0; j<2; j++)
		{
			std::cout << tp[i][j] << " ";
		}
		std::cout << std::endl;
    	}
	
	int F[2][3] = {{1, 2, 3}, {4, 5, 6}};

	double *vcm= linalg.vecmat(E,F,2);
	std::cout << "Vector Matrix Product Result:\n";
    	for (int i = 0; i < 3; i++)
    	{
		std::cout << vcm[i] << " ";
    	}
	std::cout<<std::endl;
	
	int G[3]={1,2,3};

	double *mtv= linalg.matvec(F,G,2);
	std::cout << "Matrix Vector Product Result:\n";
    	for (int i = 0; i < 2; i++)
    	{
		std::cout << mtv[i] << " ";
    	}
	std::cout<<std::endl;
	
	double H[5][5] = {
    		{25, 15, -5, 10, 20},
    		{15, 18,  0,  8, 12},
	    	{-5,  0, 11,  3,  2},
    		{10,  8,  3, 20,  5},
    		{20, 12,  2,  5, 30}};
	
	double I[3][3]= {{4,12,-16},{12,37,-43},{-16,-43,98}};
	
	double (*chl)[5]=linalg.cholesky(H);
	std::cout << "Cholesky decomposition of a Matrix Result:\n";
    	for (int i = 0; i <5; i++)
    	{
		for (int j=0; j<5; j++)
		{
			std::cout << chl[i][j] << " ";
		}
		std::cout << std::endl;
    	}
	
	double dtm=linalg.determinant(I,3);
	std::cout<<"Determinant of the Matrix"<<std::endl<<dtm<<std::endl;

	double (*inv)[3] = linalg.inverse(I, 3);
	if(inv)
	{
		std::cout << "Inverse of a Matrix Result:\n";
    		for (int i = 0; i <3; i++)
    		{
			for (int j=0; j<3; j++)
			{
				std::cout << std::setw(10) << std::fixed << std::setprecision(5) << inv[i][j] << " ";
			}
        		std::cout << "\n";
    		}
	}
    	else 
	{
        	std::cout << "\nMatrix is singular or not invertible.\n";
    	}

	auto qrdec = linalg.qrdecomp(I, 3);
    	double* Q = qrdec.first;
    	double* R = qrdec.second;
	std::cout << "QR Decompostion of a Matrix:\n";
	std::cout << "Matrix Q:\n";
    	for (int i = 0; i < 3; ++i) 
	{
        	for (int j = 0; j < 3; ++j)
            		std::cout << std::setw(10) << std::fixed << std::setprecision(5) << Q[i * 3 + j] << " ";
       		std::cout << "\n";
    	}

    	std::cout << "\nMatrix R:\n";
    	for (int i = 0; i < 3; ++i) 
	{
	        for (int j = 0; j < 3; ++j)
            		std::cout << std::setw(10) << std::fixed << std::setprecision(5) << R[i * 3 + j] << " ";
        	std::cout << "\n";
    	}

	std::vector<double> eigenvalues = linalg.eigenval(I, 3);
	std::cout << "Eigenvalues of Matrix:\n";
    	for (int i=0; i<3; i++) 
	{
        	std::cout << std::fixed << std::setprecision(10) << eigenvalues[i] << "\n";
    }
	
	std::cout << "\nSolving Ax = b using Gaussian Elimination:\n";

    std::vector<std::vector<double>> A_sys = {
        {3,4},
        {3,3}
    };

    std::vector<double> b_sys = {2, 3};
    std::vector<double> x;

    if (linalg.solveLinearSystem(A_sys, b_sys, x)) {
        std::cout << "Solution Vector x:\n";
        for (const auto& xi : x)
            std::cout << std::fixed << std::setprecision(6) << xi << " ";
        std::cout << "\n";
    } else {
        std::cout << "Failed to solve the system.\n";
    }

}
