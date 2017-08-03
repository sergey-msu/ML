using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.TextMethods.WeightingSchemes
{
  public abstract class IDFWeightingSchemeBase : IIDFWeightingScheme
  {
    public abstract string Name { get; }

    public abstract double[] GetWeights(int vocabularyCount, int[] idfFreqs);
  }

  public class UnaryIDFWeightingScheme : IDFWeightingSchemeBase
  {
    public override string Name { get { return "UNARY"; } }

    public override double[] GetWeights(int vocabularyCount, int[] idfFreqs)
    {
      var len = idfFreqs.Length;
      var result = new double[len];
      for (int i=0; i<len; i++)
        result[i] = 1;

      return result;
    }
  }

  public class StandartIDFWeightingScheme : IDFWeightingSchemeBase
  {
    public override string Name { get { return "STD"; } }

    public override double[] GetWeights(int vocabularyCount, int[] idfFreqs)
    {
      var len = idfFreqs.Length;
      var result = new double[len];
      for (int i=0; i<len; i++)
        result[i] = Math.Log(vocabularyCount/(double)idfFreqs[i]);

      return result;
    }
  }

  public class MaxIDFWeightingScheme : IDFWeightingSchemeBase
  {
    public override string Name { get { return "MAX"; } }


    public override double[] GetWeights(int vocabularyCount, int[] idfFreqs)
    {
      var max = int.MinValue;
      var len = idfFreqs.Length;
      var result = new double[len];

      for (int i=0; i<len; i++)
      {
        var freq = idfFreqs[i];
        if (max<freq) max = freq;
      }

      for (int i=0; i<len; i++)
        result[i] = Math.Log(max/(1+(double)idfFreqs[i]));

      return result;
    }
  }

  public class SmoothIDFWeightingScheme : IDFWeightingSchemeBase
  {
    public override string Name { get { return "Smooth"; } }


    public override double[] GetWeights(int vocabularyCount, int[] idfFreqs)
    {
      var len = idfFreqs.Length;
      var result = new double[len];
      for (int i=0; i<len; i++)
        result[i] = Math.Log(1 + vocabularyCount/(double)idfFreqs[i]);

      return result;
    }
  }

  public class ProbabilisticIDFWeightingScheme : IDFWeightingSchemeBase
  {
    public override string Name { get { return "PROB"; } }


    public override double[] GetWeights(int vocabularyCount, int[] idfFreqs)
    {
      var len = idfFreqs.Length;
      var result = new double[len];
      for (int i=0; i<len; i++)
      {
        var freq = idfFreqs[i];
        result[i] = Math.Log((vocabularyCount - freq)/(double)freq);
      }

      return result;
    }
  }
}
