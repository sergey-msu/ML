using System;
using ML.Contracts;
using ML.Core.Mathematics;
using ML.Core;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents convolution layer: a 4D convolution tensor kernel that accepts
  /// 3D input  - a set of 2D matrix data, and produces
  /// 3D output - a set of 2D feature maps
  ///
  /// IO:
  /// Input  - 3D array of shape (input_depth, input_height, input_width)
  /// Output - 3D array of shape (output_depth, output_height, output_width)
  ///
  /// Parameters:
  /// Kernel - 4D array of shape (output_depth, input_depth, win_size, win_size)
  /// Biases - 1D array of shape (output_depth)
  ///
  /// We use tied bias approach
  /// https://harmdevries89.wordpress.com/2015/03/27/tied-biases-vs-untied-biases/
  ///
  /// Total count of parameters = (win_size*win_size*input_length + 1) * output_length;
  /// </summary>
  public class ConvLayer : DeepLayerBase
  {
    #region Fields

    protected int m_KernelParamCount;
    protected int m_FeatureMapParamCount;
    protected int m_ParamCount;

    #endregion

    #region .ctor

    public ConvLayer(int outputDepth,
                     int windowSize,
                     int stride=1,
                     int padding=0,
                     IActivationFunction activation = null)
      : base(outputDepth,
             windowSize,
             stride,
             padding,
             activation)
    {
    }

    public ConvLayer(int outputDepth,
                     int windowHeight,
                     int windowWidth,
                     int strideHeight=1,
                     int strideWidth=1,
                     int paddingHeight=0,
                     int paddingWidth=0,
                     IActivationFunction activation = null)
      : base(outputDepth,
             windowHeight,
             windowWidth,
             strideHeight,
             strideWidth,
             paddingHeight,
             paddingWidth,
             activation)
    {
    }

    #endregion

    #region Public

    /// <summary>
    /// Tied bias value
    /// </summary>
    public double Bias(int q)
    {
      return m_Weights[(q+1)*m_FeatureMapParamCount - 1];
    }

    /// <summary>
    /// Kernal wieght value
    /// </summary>
    public double Kernel(int q, int p, int y, int x)
    {
      return m_Weights[x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount];
    }

    #endregion

    protected override void BuildParams()
    {
      m_KernelParamCount     = m_WindowHeight*m_WindowWidth;
      m_FeatureMapParamCount = m_InputDepth*m_KernelParamCount + 1;
      m_ParamCount           = m_FeatureMapParamCount*m_OutputDepth;

      m_Weights = new double[m_ParamCount];

      base.BuildParams();
    }

    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        // fm neurons
        for (int i=0; i<m_OutputHeight; i++)
        for (int j=0; j<m_OutputWidth; j++)
        {
          var net = m_Weights[(q+1)*m_FeatureMapParamCount-1]; // Bias(q)
          var xmin = j*m_StrideWidth-m_PaddingWidth;
          var ymin = i*m_StrideHeight-m_PaddingHeight;

          // window
          for (int y=0; y<m_WindowHeight; y++)
          for (int x=0; x<m_WindowWidth;  x++)
          {
            var xidx = xmin+x;
            var yidx = ymin+y;
            if (xidx>=0 && xidx<m_InputWidth && yidx>=0 && yidx<m_InputHeight)
            {
              // inner product in p-depth (over input channel's neuron at fixed position)
              for (int p=0; p<m_InputDepth; p++)
              {
                var idx = x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
                var w = m_Weights[idx]; // Kernel(q, p, y, x)
                net += w * input[p][yidx, xidx];
              }
            }
          }

          result[q][i, j] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
        }
      }
    }

    /// <summary>
    /// Backpropagate "errors" to previous layer for future use
    /// </summary>
    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
    {
      if (prevLayer==null)
        throw new MLException("Prev layer is null");

      for (int p=0; p<m_InputDepth;  p++)
      for (int i=0; i<m_InputHeight; i++)
      for (int j=0; j<m_InputWidth;  j++)
      {
        var g = 0.0D;

        for (int q=0; q<m_OutputDepth;  q++)
        for (int k=0; k<m_OutputHeight; k++)
        {
          var y = i+m_PaddingHeight-k*m_StrideHeight;
          if (y >= m_WindowHeight) continue;
          if (y < 0) break;

          for (int m=0; m<m_OutputWidth; m++)
          {
            var x = j+m_PaddingWidth-m*m_StrideWidth;
            if (x >= m_WindowWidth) continue;
            if (x < 0) break;

            var idx = x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
            g += errors[q][k, m] * m_Weights[idx]; // Kernel(q, p, y, x)
          }
        }

        var value = prevValues[p][i,j];
        var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
        prevErrors[p][i, j] = g * deriv;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
      int idx;

      // weight updates
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth;   p++)
        for (int i=0; i<m_WindowHeight; i++)
        for (int j=0; j<m_WindowWidth;  j++)
        {
          var dw = 0.0D;

          for (int k=0; k<m_OutputHeight; k++)
          {
            var y = i-m_PaddingHeight+m_StrideHeight*k;
            if (y<0) continue;
            if (y>=m_InputHeight) break;

            for (int m=0; m<m_OutputWidth; m++)
            {
              var x = j-m_PaddingWidth+m_StrideWidth*m;
              if (x<0) continue;
              if (x>=m_InputWidth) break;

              dw += errors[q][k, m] * prevValues[p][y, x];
            }
          }

          idx = j + i*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
          if (isDelta) gradient[idx] += dw; // Gradient(q, p, i, j)
          else gradient[idx] = dw;
        }

        // bias updates
        var db = 0.0D;
        for (int k=0; k<m_OutputHeight; k++)
        for (int m=0; m<m_OutputWidth; m++)
        {
          db += errors[q][k, m];
        }

        idx = (q+1)*m_FeatureMapParamCount-1;
        if (isDelta) gradient[idx] += db; // BiasGrad(q)
        else gradient[idx] = db;
      }
    }

    public override void RandomizeParameters(int seed)
    {
      var random = RandomGenerator.Get(seed);

      for (int i=0; i<m_ParamCount; i++)
      {
        m_Weights[i] = 2*random.GenerateUniform(0, 1) / m_FeatureMapParamCount;
      }
    }
  }
}
