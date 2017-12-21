//
//  main.cpp
//  c++_codes
//
//  Created by SeongMuk Gang on 2017/12/20.
//  Copyright © 2017 SeongMuk Gang. All rights reserved.
//

#include <iostream>
#include <vector>
#define WIDTH 5
#define HEIGHT 3

void matMul(int *C[], int *A[], int *B[], int heightA, int widthA, int widthB)
{
    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < widthB; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < widthA; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
                std::cout << C[i][j];
            }
        }
    }
}

void matGen2(int C[][2], int height, int width)
{
    for (int n = 0; n < height; n++){
        for (int m = 0; m < width; m++)
        {
            C[n][m] = (n + 1) * (m + 1);
            std::cout << C[n][m] << " ";
            
            //C[n*width+m] = (n + 1) * (m + 1);
            //std::cout << C[n*width+m] << " ";
        }
        std::cout << std::endl;
    }
}

void pseudoMat(int C[], int height, int width)
{
    for (int n = 0; n < height; n++){
        for (int m = 0; m < width; m++)
        {
            C[n*width+m] = (n + 1) * (m + 1);
            std::cout << C[n*width+m] << " ";
        }
        std::cout << std::endl;
    }    
}
template <class T> void printMatrix(std::vector<std::vector<T> > C)
{
    for (int i = 0; i < C.size(); i++)
    {
        for (int j = 0; j < C[i].size(); j++)
        {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "(" << C.size() << ", " 
    << C[0].size() << ")" << std::endl;
}

int main(int argc, const char *argv[])
{
    int AA[2][2];// = {{1, 1}, {2, 2}};
    matGen2(AA, 2, 2);
    std::cout << std::endl;

    int BB[2*2];
    pseudoMat(BB, 2, 2);
    std::cout << std::endl;

    std::vector<int> v2(5, 1);
    std::vector<std::vector<int> > v2d2(3, v2);
    printMatrix(v2d2);
    std::cout << std::endl;
    return 0;
}