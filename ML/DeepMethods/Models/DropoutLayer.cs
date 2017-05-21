using System;
using System.Threading;
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

    private object m_Sync = new object();

    [NonSerialized] // TODO fixup! seed must be serialized
    private int    m_Seed;
    private double m_DropRate;
    private double m_RetainRate;
    private bool   m_ApplyCustomMask;

    [NonSerialized]
    private ThreadLocal<bool[][,]> m_Mask;

    [NonSerialized]
    private ThreadLocal<RandomGenerator> m_Generator;

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

      m_Seed = seed;
      m_DropRate = rate;
      m_RetainRate = 1-rate;
    }

    #endregion

    private ThreadLocal<RandomGenerator> Generator
    {
      get
      {
        if (m_Generator==null)
        {
          lock (m_Sync)
          {
            if (m_Generator==null)
            {
              m_Generator = new ThreadLocal<RandomGenerator>(() =>
              {
                return RandomGenerator.Get(m_Seed, false);
                // WARNING:
                // RandomGenerator.Get(seed, false) used to make randomness more deterministic :)
                // "false" means that every thread will get the same instance of RandomGenerator
                // "true" or empty means that different threads will share the same RandomGenerator instance, therefore output will vary from run to run
                // WARNING #2:
                // setting value to "false" still dont makes the process fully deterministic because of undeterministic nature of threads allocation :(
              });
            }
          }
        }

        return m_Generator;
      }
    }

    private ThreadLocal<bool[][,]> ThreadMask
    {
      get
      {
        if (m_Generator==null)
        {
          lock (m_Sync)
          {
            if (m_Mask==null)
            {
              m_Mask = new ThreadLocal<bool[][,]>(() =>
              {
                if (m_IsTraining && !m_ApplyCustomMask)
                {
                  var mask = new bool[m_OutputDepth][,];
                  for (int i=0; i<m_OutputDepth; i++)
                    mask[i] = new bool[m_OutputHeight, m_OutputWidth];
                  return mask;
                }
                return null;
              });
            }
          }
        }

        return m_Mask;
      }
      set
      {

      }
    }

    #region Properties

    public override int ParamCount { get { return 0; } }

    public double DropRate { get { return m_DropRate; } }

    public double RetainRate { get { return m_RetainRate; } }

    public bool[][,] Mask
    {
      get { return ThreadMask.Value; }
      set
      {
        ThreadMask.Value=value;
        m_ApplyCustomMask = true;
      }
    }

    public bool ApplyCustomMask
    {
      get { return m_ApplyCustomMask; }
      set { m_ApplyCustomMask=value; }
    }

    #endregion

    public override void _Build()
    {
      m_ActivationFunction = null;

      m_OutputDepth  = m_InputDepth;
      m_OutputHeight = m_InputHeight;
      m_OutputWidth  = m_InputWidth;

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

      var mask = ThreadMask.Value;
      var generator = Generator;

      for (int p=0; p<m_InputDepth;  p++)
      for (int i=0; i<m_InputHeight; i++)
      for (int j=0; j<m_InputWidth;  j++)
      {
        if (m_ApplyCustomMask) // custom mask applied
        {
          result[p][i, j] = mask[p][i, j] ? input[p][i, j] / m_RetainRate : 0;
        }
        else // generate new random mask
        {
          var retain = generator.Value.Bernoulli(m_RetainRate);
          if (retain)
          {
            m_Mask.Value[p][i, j] = true;
            result[p][i, j] = input[p][i, j] / m_RetainRate;
          }
          else
          {
            m_Mask.Value[p][i, j] = false;
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
        if (m_Mask.Value[p][i, j])
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
