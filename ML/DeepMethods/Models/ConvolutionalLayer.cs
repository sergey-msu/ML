using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.DeepMethods.Models
{
  public enum BiasMode
  {
    Tied = 0,  // one bias per convolutional kernel
    Untied = 1 // one bias per kernel and output location
  }

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
    private int m_BiasParamCount;
    private int m_FeatureMapParamCount;
    private int m_ParamCount;
    private BiasMode m_BiasMode;

    private double[,,,] m_Kernel;
    private double[]    m_Biases;
    private double[,,]  m_UntiedBiases;

    #endregion

    #region .ctor

    public ConvolutionalLayer(int inputDepth,
                              int inputSize,
                              int outputDepth,
                              int windowSize,
                              int stride,
                              int padding=0,
                              BiasMode biasMode = BiasMode.Tied,
                              bool isTraining = false)
      : base(inputDepth,
             inputSize,
             outputDepth,
             windowSize,
             stride,
             padding,
             isTraining)
    {
      m_KernelParamCount     = windowSize*windowSize;
      m_BiasParamCount = (biasMode==BiasMode.Tied) ? 1 : m_OutputSize*m_OutputSize;
      m_FeatureMapParamCount = inputDepth*m_KernelParamCount + m_BiasParamCount;
      m_ParamCount           = m_FeatureMapParamCount*outputDepth;

      m_Kernel = new double[outputDepth, inputDepth, windowSize, windowSize];

      m_BiasMode = biasMode;
      if (biasMode==BiasMode.Tied)
        m_Biases = new double[outputDepth];
      else
        m_UntiedBiases = new double[outputDepth, m_OutputSize, m_OutputSize];
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_ParamCount; } }

    /// <summary>
    /// Bias mode: tied or undied
    /// </summary>
    public BiasMode BiasMode { get { return m_BiasMode; } }

    /// <summary>
    /// Tied bias values (one for each output feature map)
    /// </summary>
    public double[] Biases { get { return m_Biases; } }

    /// <summary>
    /// Untied bias values (one for each output feature map)
    /// </summary>
    public double[,,] UntiedBiases{ get { return m_UntiedBiases; } }

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

        if (m_BiasMode==BiasMode.Tied)
          m_Biases[q] = 2*random.GenerateUniform(0, 1) / coeff;
        else
        {
          for (int i=0; i<m_OutputSize; i++)
          for (int j=0; j<m_OutputSize; j++)
          {
            m_UntiedBiases[q, i, j] = 2*random.GenerateUniform(0, 1) / coeff;
          }
        }
      }
    }

    public override double[,,] Calculate(double[,,] input)
    {
      if (m_InputDepth != input.GetLength(0))
        throw new MLException("Incorrect input depth");

      MathUtils.Tensors.Convolute(input, m_Biases, m_Value, m_Kernel, m_Stride, m_Padding, ActivationFunction);

      return m_Value;
    }

    protected override double DoGetParam(int idx)
    {
      throw new NotImplementedException();
      //var fmidx  = idx / m_FeatureMapParamCount;
      //var fmpidx = idx % m_FeatureMapParamCount;
      //
      //if (fmpidx < m_FeatureMapParamCount-m_BiasParamCount) // kernel params
      //{
      //  var kidx  = fmpidx / m_KernelParamCount;
      //  var kpidx = fmpidx % m_KernelParamCount;
      //  var yidx  = kpidx / m_WindowSize;
      //  var xidx  = kpidx % m_WindowSize;
      //
      //  return m_Kernel[fmidx, kidx, yidx, xidx];
      //}
      //
      //// bias params
      //return m_Biases[fmidx];
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      throw new NotImplementedException();
      //var fmidx  = idx / m_FeatureMapParamCount;
      //var fmpidx = idx % m_FeatureMapParamCount;
      //
      //if (fmpidx == m_FeatureMapParamCount-1)
      //{
      //  if (isDelta) m_Biases[fmidx] += value;
      //  else m_Biases[fmidx] = value;
      //  return;
      //}
      //
      //var kidx  = fmpidx / m_KernelParamCount;
      //var kpidx = fmpidx % m_KernelParamCount;
      //var yidx  = kpidx / m_WindowSize;
      //var xidx  = kpidx % m_WindowSize;
      //
      //if (isDelta) m_Kernel[fmidx, kidx, yidx, xidx] += value;
      //else m_Kernel[fmidx, kidx, yidx, xidx] = value;
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      throw new NotImplementedException();
      // TODO: Do it more intelligent!!!!!
      //for (int i=0; i<m_ParamCount; i++)
      //  DoSetParam(i, pars[cursor+i], isDelta);

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
