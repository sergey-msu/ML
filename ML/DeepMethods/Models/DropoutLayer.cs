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

      m_Value = new double[m_OutputDepth][,];
      for (var q=0; q<m_OutputDepth; q++)
        m_Value[q] = new double[m_OutputHeight, m_OutputWidth];

      if (m_IsTraining && !m_ApplyCustomMask)
      {
        m_Mask = new bool[m_OutputDepth][,];
        for (int i=0; i<m_OutputDepth; i++)
          m_Mask[i] = new bool[m_OutputHeight, m_OutputWidth];
      }

      BuildParams();
    }

    protected override double[][,] DoCalculate(double[][,] input)
    {
      if (m_IsTraining)
      {
        for (int p=0; p<m_InputDepth; p++)
        for (int i=0; i<m_InputHeight;  i++)
        for (int j=0; j<m_InputWidth;  j++)
        {
          if (m_ApplyCustomMask) // custom mask applied
          {
            m_Value[p][i, j] = m_Mask[p][i, j] ? input[p][i, j] / m_RetainRate : 0;
          }
          else // generate new random mask
          {
            var retain = m_Generator.Bernoulli(m_RetainRate);
            if (retain)
            {
              m_Mask[p][i, j] = true;
              m_Value[p][i, j] = input[p][i, j] / m_RetainRate;
            }
            else
            {
              m_Mask[p][i, j] = false;
              m_Value[p][i, j] = 0;
            }
          }
        }
      }
      else
      {
        Array.Copy(input, m_Value, input.Length);
      }

      return m_Value;
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] error, double[][,] prevError)
    {
      for (int p=0; p<m_OutputDepth;  p++)
      for (int i=0; i<m_OutputHeight; i++)
      for (int j=0; j<m_OutputWidth;  j++)
      {
        prevError[p][i, j] = Mask[p][i, j] ?
                             error[p][i, j] * prevLayer.Derivative(p, i, j) / m_RetainRate :
                             0;
      }
    }

    protected override void DoSetLayerGradient(DeepLayerBase prevLayer, double[][,] errors, double[] layerGradient)
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
