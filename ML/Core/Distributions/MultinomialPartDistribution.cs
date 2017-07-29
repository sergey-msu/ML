using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Utils;

namespace ML.Core.Distributions
{
  /// <summary>
  /// Reprsents a single miltiplicative part of Multinomial distribution without the normalizing factor n!/(m1!m2!...mk!).
  /// Common Multinomial distribution:
  ///    n!/(m1!m2!...mk!) * ( p1^x1...pk^xk )
  /// The class represents only some pi^xi part
  /// </summary>
  public class MultinomialPartDistribution : DistributionBase<MultinomialPartDistribution.Parameters>
  {
    #region Inner

    public struct Parameters : IDistributionParameters
    {
      public Parameters(double p)
      {
        if (p < 0 || p > 1)
          throw new MLException("MultinomialPartDistribution+Parameters.ctor(p<0|p>1)");

        P = p;
      }

      public readonly double P;
    }

    #endregion

    private double m_Alpha;
    private int m_TotalCount;
    private int m_N;

    public MultinomialPartDistribution()
    {
    }


    /// <summary>
    /// Full sample length (e.g. length of dictionary V in a document classification task etc.)
    /// </summary>
    public int N { get { return m_N; } set { m_N=value;} }

    /// <summary>
    /// Total count of elementary events in underlying probability space (e.g. full number of tokens(words) in some class in a document classification task etc.)
    /// </summary>
    public int TotalCount
    {
      get { return m_TotalCount; }
      set
      {
        if (value <= 0)
          throw new MLException("Total count must be positive");

        m_TotalCount=value;
      }
    }

    /// <summary>
    /// Smoothing coefficient
    /// </summary>
    public double Alpha
    {
      get { return m_Alpha; }
      set
      {
        if (value<=0)
          throw new MLException("Smoothing coefficient must be positive");

        m_Alpha=value;
      }
    }

    /// <summary>
    /// If true, uses Laplace/Lidstone smoothing
    /// </summary>
    public bool UseSmoothing { get; set; }



    public override double Value(double x)
    {
      return GeneralUtils.Pow(Params.P, (int)x);
    }

    public override double LogValue(double x)
    {
      return x*Math.Log(Params.P);
    }

    public override void FromSample(double[] sample)
    {
      var a = 0.0D;
      var len = sample.Length;
      for (int i=0; i<len; i++)
        a += sample[i];
      double b = m_TotalCount;

      if (a==0 && UseSmoothing)
      {
        a += m_Alpha;
        b += m_Alpha*m_N;
      }

      Params = new Parameters(a/b);
    }

    public override Dictionary<ClassFeatureKey, Parameters> FromSample(ClassifiedSample<double[]> sample)
    {
      var result  = new Dictionary<ClassFeatureKey, Parameters>();
      var temp    = new Dictionary<ClassFeatureKey, double>();
      var dim     = sample.GetDimension();
      var classes = sample.CachedClasses;
      var ts      = new Dictionary<Class, double>();

      foreach (var cls in classes)
        ts[cls] = 0;

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          var key = new ClassFeatureKey(cls, i);

          var p = data[i];
          if (!temp.ContainsKey(key))
            temp[key] = p;
          else
            temp[key] += p;

          ts[cls] += p;
        }
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var cls in classes)
        {
          var key = new ClassFeatureKey(cls, i);
          var p = temp[key];
          var b = ts[cls];
          if (p==0 && UseSmoothing)
          {
            p += m_Alpha;
            b += m_Alpha*m_N;
          }

          result[key] = new Parameters(p/b);
        }
      }

      return result;
    }
  }
}
