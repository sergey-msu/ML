using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents convolution layer: a 4D convolution tensor kernel that accepts
  /// 3D input  - a set of 2D matrix data, and produces
  /// 3D output - a set of 2D feature maps
  ///
  /// IO:
  /// Input  - 3D array (input_length * input_size * input_size):    data[depth_idx, height_idx, width_idx]
  /// Output - 3D array (output_size * output_size * output_length): feature_maps[depth_idx, height_idx, width_idx]
  ///
  /// Parameters:
  /// Kernel - 4D array (output_length * input_length * win_size * win_size): kernel[output_idx, input_idx, height_idx, width_idx]
  /// Biases - 1D array (output_length): biases[output_idx]
  ///
  /// We use tied bias approach
  /// http://datascience.stackexchange.com/questions/17671
  /// https://harmdevries89.wordpress.com/2015/03/27/tied-biases-vs-untied-biases/
  ///
  /// Total count of parameters = (win_size * win_size * input_length + 1) * output_length;
  /// </summary>
  public class ConvolutionalLayer : DeepLayerBase
  {
    #region Fields

    private int m_KernelParamCount;
    private int m_FeatureMapParamCount;
    private int m_ParamCount;

    private double[,,,] m_Kernel;
    private double[]    m_Biases;

    #endregion

    #region .ctor

    public ConvolutionalLayer(int outputDepth,
                              int windowSize,
                              int stride=1,
                              int padding=0)
      : base(outputDepth,
             windowSize,
             stride,
             padding)
    {
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_ParamCount; } }

    /// <summary>
    /// Tied bias values (one for each output feature map)
    /// </summary>
    public double[] Biases { get { return m_Biases; } }

    /// <summary>
    /// Kernal wieght values
    /// </summary>
    public double[,,,] Kernel { get { return m_Kernel; } }

    #endregion

    /// <summary>
    /// Randomizes feature map weights
    /// </summary>
    public override void RandomizeParameters(int seed)
    {
      var random = RandomGenerator.Get(seed);
      var coeff = m_InputDepth*m_WindowSize*m_WindowSize;

      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth; p++)
        for (int i=0; i<m_WindowSize; i++)
        for (int j=0; j<m_WindowSize; j++)
        {
          m_Kernel[q, p, i, j] = 2*random.GenerateUniform(0, 1) / coeff;
        }

        m_Biases[q] = 2*random.GenerateUniform(0, 1) / coeff;
      }
    }

    public override double[,,] Calculate(double[,,] input)
    {
      if (m_InputDepth != input.GetLength(0))
        throw new MLException("Incorrect input depth");

      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        // fm neurons
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var net = m_Biases[q];
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
                net += m_Kernel[q, p, y, x]*input[p, yidx, xidx];
            }
          }

          m_Value[q, i, j] = ActivationFunction.Value(net);
        }
      }

      return m_Value;
    }

    public override void DoBuild()
    {
      base.DoBuild();

      m_KernelParamCount     = m_WindowSize*m_WindowSize;
      m_FeatureMapParamCount = m_InputDepth*m_KernelParamCount + 1;
      m_ParamCount           = m_FeatureMapParamCount*m_OutputDepth;

      m_Kernel = new double[m_OutputDepth, m_InputDepth, m_WindowSize, m_WindowSize];
      m_Biases = new double[m_OutputDepth];
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

      return m_Kernel[fmidx, kidx, yidx, xidx];
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

      if (isDelta) m_Kernel[fmidx, kidx, yidx, xidx] += value;
      else m_Kernel[fmidx, kidx, yidx, xidx] = value;
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      // TODO: Do it more intelligent
      for (int i=0; i<m_ParamCount; i++)
        DoSetParam(i, pars[cursor+i], isDelta);

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
