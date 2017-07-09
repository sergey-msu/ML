using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Utils;

namespace ML.Core.Distributions
{
  public class NormalDistribution : DistributionBase
  {
    public const double COEFF =  0.3989422804D; // 1.0D/Math.Sqrt(GeneralUtils.DOUBLE_PI);

    private double m_Theta;
    private double m_Sigma;

    public NormalDistribution(double theta = 0, double sigma = 1)
    {
      Theta = theta;
      Sigma = sigma;
    }


    public double Theta
    {
      get { return m_Theta; }
      set { m_Theta = value; }
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
      var t = (x - m_Theta)/m_Sigma;
      return COEFF/m_Sigma * Math.Exp(-t*t/2);
    }

    public override void MaximumLikelihood(double[] sample)
    {
      var theta = 0.0D;
      var n = sample.Length;
      for (int i=0; i<n; i++)
        theta += sample[i];
      theta /= n;
      m_Theta = theta;

      var sigma = 0.0D;
      for (int i=0; i<n; i++)
      {
        var t = sample[i] - theta;
        sigma += t*t;
      }
      sigma = Math.Sqrt(sigma/n);
      m_Sigma = sigma;
    }

    public override Dictionary<ClassFeatureKey, IDistribution> MaximumLikelihood(ClassifiedSample<double[]> sample)
    {
      var result = new Dictionary<ClassFeatureKey, IDistribution>();
      var dim = sample.GetDimension();
      var classes = sample.CachedClasses;

      var thetas = new Dictionary<Class, double>();
      var sigmas = new Dictionary<Class, double>();
      var mys    = new Dictionary<Class, double>();
      foreach (var cls in classes)
      {
        thetas[cls] = 0.0D;
        sigmas[cls] = 0.0D;
        mys[cls]    = 0;
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          thetas[cls] += data[i];

          if (i==0) mys[cls]++;
        }
        foreach (var cls in classes)
          thetas[cls] /= mys[cls];

        foreach (var pData in sample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          var t = data[i] - thetas[cls];
          sigmas[cls] += t*t;
        }
        foreach (var cls in classes)
        {
          var sigma = Math.Sqrt(sigmas[cls] / mys[cls]);
          result[new ClassFeatureKey(cls, i)] = new NormalDistribution(thetas[cls], sigma);
        }

        thetas.Clear();
        sigmas.Clear();
      }

      return result;
    }
  }
}
