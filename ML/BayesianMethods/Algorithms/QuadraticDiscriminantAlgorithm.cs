using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;
using ML.Core.Mathematics;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Performs quadratic discriminant analysis.
  /// </summary>
  public class QuadraticDiscriminantAlgorithm : ClassificationAlgorithmBase<double[]>
  {
    private readonly Dictionary<Class, double> m_ClassLosses;

    private Dictionary<Class, double[]>  m_Mus;
    private Dictionary<Class, double[,]> m_ISs;
    private Dictionary<Class, double>    m_Dets;
    private Dictionary<Class, int>       m_LHist;


    public QuadraticDiscriminantAlgorithm(Dictionary<Class, double> classLosses=null)
    {
      m_ClassLosses = classLosses;
    }


    public override string ID { get { return "QDISC"; } }

    public override string Name { get { return "Quadratic discriminant classification"; } }

    /// <summary>
    /// Additional multiplicative penalty to wrong object classification.
    /// If null, all class penalties dafault to 1 (no special effect on classification - pure MAP classification)
    /// </summary>
    public Dictionary<Class, double> ClassLosses { get { return m_ClassLosses; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var dim = TrainingSample.GetDimension();
      var classes = TrainingSample.CachedClasses;

      var pHist = new Dictionary<Class, double>();

      foreach (var cls in classes)
        pHist[cls] = CalculateClassScore(obj, cls);

      Class result = null;
      var max = double.MinValue;
      foreach (var score in pHist)
      {
        if (score.Value > max)
        {
          max = score.Value;
          result = score.Key;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimates closeness of given point to given classes
    /// </summary>
    public double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = TrainingSample.GetDimension();
      var my   = m_LHist[cls];
      var mu   = m_Mus[cls];
      var IS   = m_ISs[cls];
      var det  = m_Dets[cls];

      var p = 0.0D;
      for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
        p -= IS[i,j]*(obj[i] - mu[i])*(obj[j] - mu[j]);
      p /= 2;

      p -= Math.Log(det)/2;

      double penalty;
      if (m_ClassLosses == null || m_ClassLosses.TryGetValue(cls, out penalty))
        penalty = 1.0D;
      p += Math.Log(penalty*my / TrainingSample.Count);

      return p;
    }

    public void Reset()
    {
      m_Dets  = new Dictionary<Class, double>();
      m_ISs   = new Dictionary<Class, double[,]>();
      m_Mus   = new Dictionary<Class, double[]>();
      m_LHist = new Dictionary<Class, int>();
    }


    protected override void DoTrain()
    {
      // prepare

      Reset();

      var dim = TrainingSample.GetDimension();
      var classes = TrainingSample.CachedClasses;

      var Ss = new Dictionary<Class, double[,]>();

      foreach (var cls in classes)
      {
        m_Dets[cls]  = 0.0D;
        m_ISs[cls]   = new double[dim, dim];
        m_Mus[cls]   = new double[dim];
        m_LHist[cls] = 0;
        Ss[cls]      = new double[dim, dim];
      }

      // calculate class expectation vectors

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;
        var mu = m_Mus[cls];

        m_LHist[cls] += 1;

        for (int i=0; i<dim; i++)
          mu[i] += data[i];
      }

      foreach (var cls in classes)
      {
        var my = m_LHist[cls];
        var mu = m_Mus[cls];
        for (int i=0; i<dim; i++)
          mu[i] /= my;
      }

      // calculate class covariation matrices

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;
        var my   = m_LHist[cls];
        var mu   = m_Mus[cls];
        var S    = Ss[cls];

        for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          S[i,j] += (data[i]-mu[i])*(data[j] - mu[j]) / my;
      }

      foreach (var cls in classes)
      {
        var S = Ss[cls];
        var L = MatrixOps.CholeskyFactor(S);

        // calculate class covariation matrix determinants
        var s = 0.0D;
        for (int i=0; i<dim; i++)
        {
          var lii = L[i,i];
          s += lii*lii;
        }
        if (s==0 || double.IsNaN(s) || double.IsInfinity(s))
          throw new MLException("Unable to inverse covariation matrix due to ill conditioned. Try reduce dimemsion of feature space");
        m_Dets[cls] = s;

        var IL = MatrixOps.LowerTriangelInverce(L);
        m_ISs[cls] = MatrixOps.Mult(MatrixOps.Transpose(IL), IL);
      }
    }

  }
}
