using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Utils;

namespace ML.Core.Distributions
{
  public class NormalDistribution : DistributionBase<NormalDistribution.Parameters>
  {
    #region Inner

    public struct Parameters : IDistributionParameters
    {
      public Parameters(double mu = 0, double sigma = 1)
      {
        if (sigma <= 0)
          throw new MLException("NormalDistribution+Parameters.ctor(sigma<=0)");

        Mu = mu;
        Sigma = sigma;
      }

      public readonly double Mu;
      public readonly double Sigma;
    }

    #endregion

    public const double COEFF =  0.3989422804D; // 1.0D/Math.Sqrt(GeneralUtils.DOUBLE_PI);

    private double m_Mu;
    private double m_Sigma;

    public NormalDistribution(double mu = 0, double sigma = 1)
    {
      Mu = mu;
      Sigma = sigma;
    }


    public double Mu
    {
      get { return m_Mu; }
      set { m_Mu = value; }
    }

    public double Sigma
    {
      get { return m_Sigma; }
      set
      {
        if (value <= 0)
          throw new MLException("NormalDistribution.Sigma(value<=0)");

        m_Sigma = value;
      }
    }

    /// <summary>
    /// Returns value of probalility (in the case of discrete distribution)
    /// or probability density (in the case of continuous distribution)
    /// </summary>
    public override double Value(double x)
    {
      var t = (x - m_Mu)/m_Sigma;
      return COEFF/m_Sigma * Math.Exp(-t*t/2);
    }

    public override void MaximumLikelihood(double[] sample)
    {
      var mu = 0.0D;
      var n = sample.Length;
      for (int i=0; i<n; i++)
        mu += sample[i];
      mu /= n;
      m_Mu = mu;

      var sigma = 0.0D;
      for (int i=0; i<n; i++)
      {
        var t = sample[i] - mu;
        sigma += t*t;
      }
      sigma = Math.Sqrt(sigma/n);
      m_Sigma = sigma;
    }

    public override Dictionary<ClassFeatureKey, Parameters> MaximumLikelihood(ClassifiedSample<double[]> sample)
    {
      var result  = new Dictionary<ClassFeatureKey, Parameters>();
      var dim     = sample.GetDimension();
      var classes = sample.CachedClasses;

      var mus    = new Dictionary<Class, double>();
      var sigmas = new Dictionary<Class, double>();
      var mys    = new Dictionary<Class, double>();
      foreach (var cls in classes)
      {
        mus[cls] = 0.0D;
        sigmas[cls] = 0.0D;
        mys[cls]    = 0;
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
          mus[cls] /= mys[cls];

        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          var t = data[i] - mus[cls];
          sigmas[cls] += t*t;
        }

        foreach (var cls in classes)
        {
          var sigma = Math.Sqrt(sigmas[cls] / mys[cls]);
          result[new ClassFeatureKey(cls, i)] = new Parameters(mus[cls], sigma);
          mus[cls] = 0.0D;
          sigmas[cls] = 0.0D;
        }
      }

      return result;
    }
  }
}
