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

      if (UseSmoothing)
      {
        a += m_Alpha;
        b += m_Alpha*m_N;
      }

      Params = new Parameters(a/b);
    }

    public override Parameters[][] FromSample(ClassifiedSample<double[]> sample)
    {
      var dim     = sample.GetDimension();
      var classes = sample.CachedClasses;
      var ts      = new double[classes.Count];
      var result  = new Parameters[classes.Count][];
      var temp    = new double[classes.Count][];
      foreach (var cls in classes)
      {
        result[cls.Value] = new Parameters[dim];
        temp[cls.Value]   = new double[dim];
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;

          var p = data[i];
          temp[cls.Value][i] += p;

          ts[cls.Value] += p;
        }
      }

      foreach (var cls in classes)
      {
        var tmps = temp[cls.Value];
        var rs   = result[cls.Value];
        var bs   = ts[cls.Value];
        if (UseSmoothing) bs += m_Alpha*m_N;

        for (int i=0; i<dim; i++)
        {
          var p = tmps[i];
          if (UseSmoothing) p += m_Alpha;

          rs[i] = new Parameters(p/bs);
        }
      }

      return result;
    }
  }
}
