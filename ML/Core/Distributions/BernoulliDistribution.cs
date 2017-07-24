using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Utils;

namespace ML.Core.Distributions
{
  public class BernoulliDistribution : DistributionBase<BernoulliDistribution.Parameters>
  {
    #region Inner

    public struct Parameters : IDistributionParameters
    {
      public Parameters(double p)
      {
        if (p < 0 || p > 1)
          throw new MLException("BernoulliDistribution+Parameters.ctor(p<0|p>1)");

        P = p;
      }

      public readonly double P;
    }

    #endregion

    public BernoulliDistribution()
    {
    }

    /// <summary>
    /// Returns Bernoulli probability density
    /// </summary>
    public override double Value(double x)
    {
      if (x==0) return 1-Params.P;
      if (x==1) return Params.P;

      throw new MLException("Bernoulli distribution input is out of range {0, 1}");
    }

    /// <summary>
    /// Returns logarithmic value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public override double LogValue(double x)
    {
      if (x==0) return Math.Log(1-Params.P);
      if (x==1) return Math.Log(Params.P);

      throw new MLException("Bernoulli distribution input is out of range {0, 1}");
    }

    public override void FromSample(double[] sample)
    {
      var mu = 0.0D;
      var n = sample.Length;
      for (int i=0; i<n; i++)
        mu += sample[i];
      mu /= n;

      Params = new Parameters(mu);
    }

    public override Dictionary<ClassFeatureKey, Parameters> FromSample(ClassifiedSample<double[]> sample)
    {
      var result  = new Dictionary<ClassFeatureKey, Parameters>();
      var dim     = sample.GetDimension();
      var classes = sample.CachedClasses;
      var mus     = new Dictionary<Class, double>();
      var mys     = new Dictionary<Class, double>();

      foreach (var cls in classes)
      {
        mus[cls] = 0.0D;
        mys[cls] = 0;
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          mus[cls] += data[i];

          if (i==0) mys[cls]++;
        }

        foreach (var cls in classes)
        {
          var mu = mus[cls]/mys[cls];
          result[new ClassFeatureKey(cls, i)] = new Parameters(mu);
          mus[cls] = 0.0D;
        }
      }

      return result;
    }
  }
}
