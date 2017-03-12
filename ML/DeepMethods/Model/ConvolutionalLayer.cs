using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents convolution layer: a 4D convolution tensor kernel that accepts
  /// 3D input  - a set of 2D matrix data, and produces
  /// 3D output - a set of 2D feature maps
  ///
  /// IO:
  /// Input  - 3D array (input_size * input_size * input_length):    data[height_idx, width_idx, depth_idx]
  /// Output - 3D array (output_size * output_size * output_length): feature_maps[height_idx, width_idx, depth_idx]
  ///
  /// Parameters:
  /// Kernel - 4D array (win_size * win_size * input_length * output_length): kernel[height_idx, width_idx, input_idx, output_idx]
  /// Biases - 1D array (output_length): biases[output_idx]
  ///
  /// Total count of parameters = (win_size * win_size * input_length + 1) * output_length;
  /// </summary>
  public class ConvolutionalLayer : DeepLayerBase
  {
    #region Fields

    private int m_WindowSize;
    private int m_Stride;
    private int m_Padding;

    private int m_KernelParamCount;
    private int m_FeatureMapParamCount;
    private int m_ParamCount;

    private double[,,,] m_Kernel;
    private double[]    m_Biases;
    private double[,,]  m_NetValue;
    private double[,,]  m_Value;
    private double[,,]  m_Derivative;

    #endregion

    #region .ctor

    public ConvolutionalLayer(int inputDepth,
                              int inputSize,
                              int outputDepth,
                              int windowSize,
                              int stride=1,
                              int padding=0)
      : base(inputDepth, inputSize, outputDepth, (inputSize - windowSize + 2*padding)/stride + 1)
    {
      if (windowSize <= 0)
        throw new MLException("ConvolutionalLayer.ctor(windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("ConvolutionalLayer.ctor(windowSize>inputSize)");
      if (stride < 0)
        throw new MLException("ConvolutionalLayer.ctor(stride<0)");
      if (padding < 0)
        throw new MLException("ConvolutionalLayer.ctor(padding<0)");

      m_WindowSize  = windowSize;
      m_Stride      = stride;
      m_Padding     = padding;

      m_KernelParamCount     = windowSize*windowSize;
      m_FeatureMapParamCount = inputDepth*m_KernelParamCount + 1;
      m_ParamCount           = m_FeatureMapParamCount*outputDepth;

      m_Kernel     = new double[windowSize, windowSize, inputDepth, outputDepth];
      m_Biases     = new double[outputDepth];
      m_Value      = new double[m_OutputSize, m_OutputSize, m_OutputDepth];
      m_NetValue   = new double[m_OutputSize, m_OutputSize, m_OutputDepth];
      m_Derivative = new double[m_OutputSize, m_OutputSize, m_OutputDepth];
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_ParamCount; } }

    /// <summary>
    /// Size of square convolution window
    /// </summary>
    public int WindowSize { get { return m_WindowSize; } }

    /// <summary>
    /// An overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// </summary>
    public int Stride { get { return m_Stride; } }

    /// <summary>
    /// Padding of the input channel
    /// </summary>
    public int Padding { get { return m_Padding; } }

    #endregion

    /// <summary>
    /// Randomizes feature map weights
    /// </summary>
    public override void RandomizeParameters(int seed)
    {
      var random = RandomGenerator.Get(seed);

      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth; p++)
        for (int i=0; i<m_WindowSize; i++)
        for (int j=0; j<m_WindowSize; j++)
        {
          m_Kernel[i, j, p, q] = 2*random.GenerateUniform(0, 1) / m_ParamCount;
        }

        m_Biases[q] = 2*random.GenerateUniform(0, 1) / m_ParamCount;
      }
    }

    public override double[,,] Calculate(double[,,] input)
    {
      if (m_InputDepth != input.GetLength(2))
        throw new MLException("Incorrect input depth");

      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        var net = m_Biases[q];

        // fm neurons
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var xmin = j*m_Stride-m_Padding;
          var ymin = i*m_Stride-m_Padding;

          // window
          for (int y=0; y<m_WindowSize; y++)
          for (int x=0; x<m_WindowSize; x++)
          {
            var xidx = xmin+x;
            var yidx = ymin+y;
            if (xidx>=0 && xidx<m_InputSize && yidx>=0 && yidx<m_InputSize)
            {
              // inner product in depth (over input channel's neuron at fixed position)
              for (int p=0; p<m_InputDepth; p++)
                net += m_Kernel[y, x, p, q]*input[yidx, xidx, p];
            }
          }

          m_NetValue[i, j, q]   = net;
          m_Derivative[i, j, q] = ActivationFunction.Derivative(net);
          m_Value[i, j, q]      = ActivationFunction.Value(net);
        }
      }

      return m_Value;
    }

    protected override double DoGetParam(int idx)
    {
      var fmidx  = idx / m_FeatureMapParamCount;
      var fmpidx = idx % m_FeatureMapParamCount;

      if (fmpidx == m_FeatureMapParamCount-1)
        return m_Biases[fmidx];

      var kidx  = fmpidx / m_KernelParamCount;
      var kpidx = fmpidx % m_KernelParamCount;
      var yidx  = kpidx / m_WindowSize;
      var xidx  = kpidx % m_WindowSize;

      return m_Kernel[yidx, xidx, kidx, fmidx];
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      var fmidx  = idx / m_FeatureMapParamCount;
      var fmpidx = idx % m_FeatureMapParamCount;

      if (fmpidx == m_FeatureMapParamCount-1)
      {
        if (isDelta) m_Biases[fmidx] += value;
        else m_Biases[fmidx] = value;
        return;
      }

      var kidx  = fmpidx / m_KernelParamCount;
      var kpidx = fmpidx % m_KernelParamCount;
      var yidx  = kpidx / m_WindowSize;
      var xidx  = kpidx % m_WindowSize;

      if (isDelta) m_Kernel[yidx, xidx, kidx, fmidx] += value;
      else m_Kernel[yidx, xidx, kidx, fmidx] = value;
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      for (int i=0; i<m_ParamCount; i++)
        DoSetParam(i, pars[cursor+i], isDelta);

      // TODO: Do it more intelligent!!!!!

      //var fmidx  = cursor / m_FeatureMapParamCount;
      //var fmpidx = cursor % m_FeatureMapParamCount;
      //var kidx   = 0;
      //var kpidx  = 0;
      //var yidx   = 0;
      //var xidx   = 0;
      //
      //if (fmpidx != m_FeatureMapParamCount-1)
      //{
      //  kidx  = fmpidx / m_KernelParamCount;
      //  kpidx = fmpidx % m_KernelParamCount;
      //  yidx  = kpidx / m_WindowSize;
      //  xidx  = kpidx % m_WindowSize;
      //}
      //
      //for (int i=0; i<m_ParamCount; i++)
      //{
      //  if (fmpidx == m_FeatureMapParamCount-1)
      //  {
      //    if (isDelta) m_Biases[fmidx] += pars[cursor++];
      //    else m_Biases[fmidx] = pars[cursor++];
      //
      //    xidx++;
      //    if (xidx==m_WindowSize) { yidx++; xidx=0; }
      //    if (yidx==m_WindowSize) { kidx++; yidx=0; }
      //    if (kidx==)
      //
      //    continue;
      //  }
      //
      //  if (isDelta)
      //    m_Kernel[yidx, xidx, kidx, fmidx] += value;
      //  else
      //    m_Kernel[yidx, xidx, kidx, fmidx] = value;
      //}



    }
  }
}
