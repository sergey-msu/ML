using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Naive Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public abstract class BayesianAlgorithmBase : ClassificationAlgorithmBase<double[]>, IGammaAlgorithm<double[]>
  {
    private readonly double[] m_ClassLosses;

    private double[] m_PriorProbs;
    private int[] m_ClassHist;
    private int m_DataDim;
    private int m_DataCount;


    protected BayesianAlgorithmBase(double[] classLosses=null)
    {
      m_ClassLosses = classLosses;
    }


    /// <summary>
    /// Additional multiplicative penalty to wrong object classification.
    /// If null, all class penalties dafault to 1 (no special effect on classification - pure MAP classification)
    /// </summary>
    public double[] ClassLosses { get { return m_ClassLosses; } }

    /// <summary>
    /// Prior class logarithm pobabilities
    /// </summary>
    public double[] PriorProbs { get { return m_PriorProbs; } }
    public int[]    ClassHist  { get { return m_ClassHist; } }
    public int      DataDim    { get { return m_DataDim; } }
    public int      DataCount  { get { return m_DataCount; } }

    /// <summary>
    /// Caclulates object proximity to some class
    /// </summary>
    public abstract double CalculateClassScore(double[] obj, Class cls);


    protected override void DoTrain()
    {
      base.DoTrain();

      var classes = Classes.ToList();
      for (int i=0; i<classes.Count; i++)
      {
        var any = classes.Any(c => (int)c.Value==i);
        if (!any)  throw new MLException(string.Format("Class values must be enumerated from 0 to {0}", classes.Count));
      }

      m_ClassHist  = new int[classes.Count];
      m_PriorProbs = new double[classes.Count];
      m_DataCount  = TrainingSample.Count;
      m_DataDim    = TrainingSample.GetDimension();

      foreach (var pData in TrainingSample)
      {
        var cls = pData.Value;
        m_ClassHist[cls.Value] += 1;
      }

      foreach (var cls in classes)
      {
        var penalty = (ClassLosses == null) ? 1 : ClassLosses[cls.Value];
        m_PriorProbs[cls.Value] = Math.Log(penalty*m_ClassHist[cls.Value]/(double)m_DataCount);
      }

      TrainImpl();
    }

    protected abstract void TrainImpl();
  }

  /// <summary>
  /// Naive Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public abstract class BayesianKernelAlgorithmBase : BayesianAlgorithmBase, IKernelAlgorithm<double[]>
  {
    private double m_H;
    private readonly IKernel m_Kernel;

    protected BayesianKernelAlgorithmBase(IKernel kernel, double h, double[] classLosses=null)
      : base(classLosses)
    {
      if (kernel == null)
        throw new MLException("BayesianKernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
      H = h;
    }


    /// <summary>
    /// Kernel function
    /// </summary>
    public IKernel Kernel { get { return m_Kernel; } }

    /// <summary>
    /// Window width
    /// </summary>
    public double H
    {
     get { return m_H; }
     set
     {
       if (value <= double.Epsilon)
         throw new MLException("BayesianKernelAlgorithmBase.H(value<=0)");

       m_H = value;
     }
    }

    public bool UseKernelMinValue { get; set; }

    public double KernelMinValue { get; set; }


    protected override void TrainImpl()
    {
      // no special training needed in kernel methods by default
    }
  }
}
