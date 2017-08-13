using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Performs quadratic discriminant analysis.
  /// </summary>
  public class QuadraticDiscriminantAlgorithm : BayesianAlgorithmBase
  {
    private readonly Dictionary<Class, double> m_ClassLosses;

    private double[][]  m_Mus;
    private double[][,] m_ISs;
    private double[]    m_Dets;


    public QuadraticDiscriminantAlgorithm(Dictionary<Class, double> classLosses=null)
    {
      m_ClassLosses = classLosses;
    }


    public override string Name { get { return "QDISC"; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(double[] obj, int cnt)
    {
      var classes = Classes;
      var scores = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var p = CalculateClassScore(obj, cls);
        scores.Add(new ClassScore(cls, p));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    /// <summary>
    /// Estimates closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = DataDim;
      var mu  = m_Mus[cls.Value];
      var IS  = m_ISs[cls.Value];
      var det = m_Dets[cls.Value];

      var p = 0.0D;
      for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
        p -= IS[i,j]*(obj[i] - mu[i])*(obj[j] - mu[j]);

      p /= 2;
      p += PriorProbs[cls.Value] - Math.Log(det)/2;

      return p;
    }

    public void Reset()
    {
      m_Dets  = new double[Classes.Length];
      m_ISs   = new double[Classes.Length][,];
      m_Mus   = new double[Classes.Length][];

      foreach (var cls in Classes)
      {
        m_ISs[cls.Value]  = new double[DataDim, DataDim];
        m_Mus[cls.Value]  = new double[DataDim];
      }
    }


    protected override void TrainImpl()
    {
      // prepare

      Reset();

      var dim = DataDim;
      var classes = Classes;
      var Ss = new double[classes.Length][,];
      foreach (var cls in Classes)
        Ss[cls.Value]  = new double[dim, dim];

      // calculate class expectation vectors

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;
        var mu = m_Mus[cls.Value];

        ClassHist[cls.Value] += 1;

        for (int i=0; i<dim; i++)
          mu[i] += data[i];
      }

      foreach (var cls in classes)
      {
        var my = ClassHist[cls.Value];
        var mu = m_Mus[cls.Value];
        for (int i=0; i<dim; i++)
          mu[i] /= my;
      }

      // calculate class covariation matrices

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;
        var my   = ClassHist[cls.Value];
        var mu   = m_Mus[cls.Value];
        var S    = Ss[cls.Value];

        for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          S[i,j] += (data[i]-mu[i])*(data[j] - mu[j]) / my;
      }

      foreach (var cls in classes)
      {
        var S = Ss[cls.Value];
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
        m_Dets[cls.Value] = s;

        var IL = MatrixOps.LowerTriangelInverce(L);
        m_ISs[cls.Value] = MatrixOps.Mult(MatrixOps.Transpose(IL), IL);
      }
    }

  }
}
