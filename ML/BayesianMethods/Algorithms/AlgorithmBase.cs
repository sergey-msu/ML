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
    private readonly Dictionary<Class, double> m_ClassLosses;

    private Dictionary<Class, double> m_PriorProbs;
    private Dictionary<Class, int> m_ClassHist;
    private int m_DataDim;
    private int m_DataCount;
    private List<Class> m_DataClasses;


    protected BayesianAlgorithmBase(Dictionary<Class, double> classLosses=null)
    {
      m_ClassLosses = classLosses;
    }


    /// <summary>
    /// Additional multiplicative penalty to wrong object classification.
    /// If null, all class penalties dafault to 1 (no special effect on classification - pure MAP classification)
    /// </summary>
    public Dictionary<Class, double> ClassLosses { get { return m_ClassLosses; } }

    /// <summary>
    /// Prior class logarithm pobabilities
    /// </summary>
    public Dictionary<Class, double> PriorProbs { get { return m_PriorProbs; } }
    public Dictionary<Class, int>    ClassHist  { get { return m_ClassHist; } }
    public int DataDim             { get { return m_DataDim; } }
    public int DataCount           { get { return m_DataCount; } }
    public List<Class> DataClasses { get { return m_DataClasses; } }

    /// <summary>
    /// Caclulates object proximity to some class
    /// </summary>
    public abstract double CalculateClassScore(double[] obj, Class cls);


    protected override void DoTrain()
    {
      m_ClassHist   = new Dictionary<Class, int>();
      m_PriorProbs  = new Dictionary<Class, double>();
      m_DataCount   = TrainingSample.Count;
      m_DataDim     = TrainingSample.GetDimension();
      m_DataClasses = TrainingSample.Classes.ToList();

      foreach (var pData in TrainingSample)
      {
        var cls = pData.Value;
        if (!m_ClassHist.ContainsKey(cls)) m_ClassHist[cls] = 1;
        else m_ClassHist[cls] += 1;
      }

      foreach (var cls in m_DataClasses)
      {
        double penalty;
        if (ClassLosses == null || !ClassLosses.TryGetValue(cls, out penalty)) penalty = 1;
        m_PriorProbs[cls] = Math.Log(penalty*m_ClassHist[cls]/(double)m_DataCount);
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

    protected BayesianKernelAlgorithmBase(IKernel kernel, double h, Dictionary<Class, double> classLosses=null)
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


    protected override void TrainImpl()
    {
      // no special training needed in kernel methods by default
    }
  }
}
