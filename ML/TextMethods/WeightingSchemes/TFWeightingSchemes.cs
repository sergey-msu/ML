using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.TextMethods.WeightingSchemes
{
  public abstract class TFWeightingSchemeBase : ITFWeightingScheme
  {
    public abstract string Name { get; }

    public abstract double GetFrequency(double[] initFreqs, int idx);

    public virtual void Reset()
    {
    }
  }

  public class BinaryTFWeightingScheme : TFWeightingSchemeBase
  {
    public override string Name { get { return "BINARY"; } }

    public override double GetFrequency(double[] initFreqs, int idx)
    {
      return (initFreqs[idx]>0) ? 1 : 0;
    }
  }

  public class RawCountTFWeightingScheme : TFWeightingSchemeBase
  {
    public override string Name { get { return "ROW_COUNT"; } }

    public override double GetFrequency(double[] initFreqs, int idx)
    {
      return initFreqs[idx];
    }
  }

  public class TermFrequencyTFWeightingScheme : TFWeightingSchemeBase
  {
    private double m_Sum;

    public TermFrequencyTFWeightingScheme()
    {
      Reset();
    }

    public override string Name { get { return "TERM_FREQ"; } }

    public override double GetFrequency(double[] initFreqs, int idx)
    {
      if (m_Sum < 0)
      {
        m_Sum = 0;
        var len = initFreqs.Length;
        for (int i=0; i<len; i++)
          m_Sum += initFreqs[i];
      }

      return initFreqs[idx]/(double)m_Sum;
    }

    public override void Reset()
    {
      m_Sum = -1;
    }
  }

  public class LogNormalizationTFWeightingScheme : TFWeightingSchemeBase
  {
    public override string Name { get { return "LOG_NORM"; } }

    public override double GetFrequency(double[] initFreqs, int idx)
    {
      var initFreq = initFreqs[idx];
      return (initFreq>0) ? 1+Math.Log(initFreq) : 0;
    }
  }

  public class DoubleNormalizationTFWeightingScheme : TFWeightingSchemeBase
  {
    private double m_MaxFreq;

    public DoubleNormalizationTFWeightingScheme(double k=0.5D)
    {
      if (k<=0) throw new MLException("DoubleNormalizationTFWeightingScheme.ctor(k<=0)");
        K = k;

      Reset();
    }

    public readonly double K;

    public override string Name { get { return "DOUBLE_NORM"; } }


    public override double GetFrequency(double[] initFreqs, int idx)
    {
      if (m_MaxFreq<0)
      {
        var len = initFreqs.Length;
        for (int i=0; i<len; i++)
        {
          var freq = initFreqs[i];
          if (m_MaxFreq<freq) m_MaxFreq = freq;
         }
      }

      return K + (1-K)*initFreqs[idx]/m_MaxFreq;
    }

    public override void Reset()
    {
      m_MaxFreq = -1;
    }
  }
}
