using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Base class for algorithm that accepts whole traing sample
  /// </summary>
  public abstract class AlgorithmBase<TParam> : IAlgorithm
  {
    private readonly ClassifiedSample  m_TrainingSample;
    private readonly Dictionary<string, Class> m_Classes;
    private readonly TParam m_Parameters;

    protected AlgorithmBase(ClassifiedSample classifiedSample, TParam pars)
    {
      if (classifiedSample == null || !classifiedSample.Any())
        throw new ArgumentException("AlrogithmBase.ctor(classifiedSample=null|empty)");

      m_TrainingSample = new ClassifiedSample(classifiedSample);
      m_Classes = m_TrainingSample.Classes.ToDictionary(c => c.Name);
      m_Parameters = pars;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public abstract string ID { get; }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Training sample
    /// </summary>
    public ClassifiedSample TrainingSample { get { return m_TrainingSample; } }

    /// <summary>
    /// Known classes
    /// </summary>
    public Dictionary<string, Class> Classes { get { return m_Classes; } }

    /// <summary>
    /// Algorithm parameters
    /// </summary>
    public TParam Parameters { get { return m_Parameters; } }

    /// <summary>
    /// Classify point
    /// </summary>
    public abstract Class Classify(Point x);
  }
}
