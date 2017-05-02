using System;
using ML.Core;
using ML.Core.Mathematics;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents dropout layer (see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  /// Inverted dropout is used instead of vanilla dropout (see http://cs231n.github.io/neural-networks-2)
  /// </summary>
  public class DropoutLayer : DeepLayerBase
  {
    #region Fields

    private double    m_DropRate;
    private double    m_RetainRate;
    private bool[][,] m_Mask;
    private bool      m_ApplyCustomMask;

    private RandomGenerator m_Generator;

    #endregion

    #region .ctor

    public DropoutLayer(double rate, int seed = 0)
      : base(outputDepth: 1, // will be overridden with input depth when building the layer
             windowSize: 1,
             stride : 1,
             padding : 0,
             activation: null)
    {
      if (rate<=0 || rate>=1)
        throw new MLException("Incorrect dropout rate");

      m_DropRate = rate;
      m_RetainRate = 1-rate;
      m_Generator = RandomGenerator.Get(seed);
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

    public double DropRate { get { return m_DropRate; } }

    public double RetainRate { get { return m_RetainRate; } }

    public bool[][,] Mask
    {
      get { return m_Mask; }
      set
      {
        m_Mask=value;
        m_ApplyCustomMask = true;
      }
    }

    #endregion

    public override void _Build()
    {
      m_ActivationFunction = null;

      m_OutputDepth  = m_InputDepth;
      m_OutputHeight = m_InputHeight;
      m_OutputWidth  = m_InputWidth;

      if (m_IsTraining && !m_ApplyCustomMask)
      {
        m_Mask = new bool[m_OutputDepth][,];
        for (int i=0; i<m_OutputDepth; i++)
          m_Mask[i] = new bool[m_OutputHeight, m_OutputWidth];
      }

      BuildParams();
    }

    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      if (!m_IsTraining)
      {
        for (int i=0; i<input.Length; i++)
          Array.Copy(input[i], result[i], input[i].Length);
        return;
      }

      for (int p=0; p<m_InputDepth;  p++)
      for (int i=0; i<m_InputHeight; i++)
      for (int j=0; j<m_InputWidth;  j++)
      {
        if (m_ApplyCustomMask) // custom mask applied
        {
          result[p][i, j] = m_Mask[p][i, j] ? input[p][i, j] / m_RetainRate : 0;
        }
        else // generate new random mask
        {
          var retain = m_Generator.Bernoulli(m_RetainRate);
          if (retain)
          {
            m_Mask[p][i, j] = true;
            result[p][i, j] = input[p][i, j] / m_RetainRate;
          }
          else
          {
            m_Mask[p][i, j] = false;
            result[p][i, j] = 0;
          }
        }
      }
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
    {
      for (int p=0; p<m_OutputDepth;  p++)
      for (int i=0; i<m_OutputHeight; i++)
      for (int j=0; j<m_OutputWidth;  j++)
      {
        if (Mask[p][i, j])
        {
          var value = prevValues[p][i, j];
          var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
          prevErrors[p][i, j] = errors[p][i, j] * deriv / m_RetainRate;
        }
        else
          prevErrors[p][i, j] = 0;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
    }


    protected override double DoGetParam(int idx)
    {
      return 0;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
    }
  }
}
