using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
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

    public abstract string ID { get; }
    public abstract string Name { get; }
    public ClassifiedSample TrainingSample { get { return m_TrainingSample; } }
    public Dictionary<string, Class> Classes { get { return m_Classes; } }
    public TParam Parameters { get { return m_Parameters; } }

    public Class Classify(Point x)
    {
      Class result = null;
      var maxEst = float.MinValue;

      foreach (var cls in m_Classes.Values)
      {
        var est = EstimateClose(x, cls);
        if (est > maxEst)
        {
          maxEst = est;
          result = cls;
        }
      }

      return result;
    }

    public abstract float EstimateClose(Point point, Class cls);
  }
}
