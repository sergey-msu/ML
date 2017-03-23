using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ML.Contracts;

namespace ML.Core.Mathematics
{
  public static partial class MathUtils
  {
    /// <summary>
    /// Utilitary Tensor functions
    /// </summary>
    public static class Tensors
    {
      /// <summary>
      /// Caclulates convolution of input 3D tensor with some kernel function
      /// </summary>
      /// <param name="input">Input 3D tensor</param>
      /// <param name="shifts">Shifts one to each output layer to add liner non-homogeneity to convolution operation</param>
      /// <param name="output">Output 3D tensor to write the result in</param>
      /// <param name="kernel">Convolution kernel (dot product coefficients)</param>
      /// <param name="stride">Convolution window step</param>
      /// <param name="padding">Covolution window padding</param>
      /// <param name="activation">After-calculation activation transform</param>
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public static void Convolute(double[,,]  input,
                                   double[]    shifts,
                                   double[,,]  output,
                                   double[,,,] kernel,
                                   int stride  = 1,
                                   int padding = 0,
                                   IFunction activation = null)
      {
        var inputDepth   = input.GetLength(0);
        var inputHeight  = input.GetLength(1);
        var inputWidth   = input.GetLength(2);
        var outputDepth  = output.GetLength(0);
        var outputHeight = output.GetLength(1);
        var outputWidth  = output.GetLength(2);
        activation = activation ?? Registry.ActivationFunctions.Identity;

        // output fm-s
        for (int q=0; q<outputDepth; q++)
        {
          // fm neurons
          for (int i=0; i<outputHeight; i++)
          for (int j=0; j<outputWidth;  j++)
          {
            var net = shifts[q];
            var xmin = j*stride-padding;
            var ymin = i*stride-padding;

            // window
            for (int y=0; y<outputHeight;  y++)
            for (int x=0; x<outputWidth; x++)
            {
              var xidx = xmin+x;
              var yidx = ymin+y;
              if (xidx>=0 && xidx<inputWidth && yidx>=0 && yidx<inputHeight)
              {
                // inner product in depth (over input channel's neuron at fixed position)
                for (int p=0; p<inputDepth; p++)
                  net += kernel[q, p, y, x]*input[p, yidx, xidx];
              }
            }

            output[q, i, j] = activation.Value(net);
          }
        }
      }


    }
  }
}
