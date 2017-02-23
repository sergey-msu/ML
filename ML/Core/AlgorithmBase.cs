using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Base class for algorithm that accepts whole traing sample
  /// </summary>
  public abstract class AlgorithmBase : IAlgorithm
  {
    #region Inner

      public delegate bool SampleMaskDelegate(Point p, Class c, int i);

      public class MaskHandle : IDisposable
      {
        private static readonly SampleMaskDelegate ALL = (p, c, i) => true;

        private AlgorithmBase m_Algorithm;
        private SampleMaskDelegate m_Mask;
        private ClassifiedSample m_MaskedSample;
        private bool m_Disposed;

        public MaskHandle(AlgorithmBase algorithm, SampleMaskDelegate mask)
        {
          if (algorithm==null)
            throw new MLException("MaskHandle.ctor(algorithm=null)");

          m_Algorithm = algorithm;
          m_Mask = mask ?? ALL;
        }

        public ClassifiedSample MaskedSample
        {
          get
          {
            if (m_Disposed) throw new ObjectDisposedException("MaskHandle has been disposed");

            if (m_MaskedSample == null)
            {
              var maskedSample = m_Algorithm.m_TrainingSample
                                            .Where((kvp, i) => m_Mask(kvp.Key, kvp.Value, i))
                                            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

              m_MaskedSample = new ClassifiedSample(maskedSample);
            }

            return m_MaskedSample;
          }
        }

        public void Dispose()
        {
          m_Algorithm.m_MaskHandle = null;
          m_Algorithm = null;
          m_Disposed = true;
        }
      }

      /// <summary>
      /// Represents classification error
      /// </summary>
      public class Error
      {
        public Error(Point point, Class realClass, Class calcClass)
        {
          Opject = point;
          RealClass = realClass;
          CalcClass = calcClass;
        }

        /// <summary>
        /// Classified object
        /// </summary>
        public readonly Point Opject;

        /// <summary>
        /// Real point class
        /// </summary>
        public readonly Class RealClass;

        /// <summary>
        /// Calculated oblect class
        /// </summary>
        public readonly Class CalcClass;
      }

    #endregion

    private readonly ClassifiedSample m_TrainingSample;
    private readonly Dictionary<string, Class> m_Classes;
    private MaskHandle m_MaskHandle;

    protected AlgorithmBase(ClassifiedSample classifiedSample)
    {
      if (classifiedSample == null || !classifiedSample.Any())
        throw new MLException("AlrogithmBase.ctor(classifiedSample=null|empty)");

      m_TrainingSample = new ClassifiedSample(classifiedSample);
      m_Classes = m_TrainingSample.Classes.ToDictionary(c => c.Name);
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
    public ClassifiedSample TrainingSample
    {
      get
      {
        return m_MaskHandle != null ? m_MaskHandle.MaskedSample : m_TrainingSample;
      }
    }

    /// <summary>
    /// Known classes
    /// </summary>
    public Dictionary<string, Class> Classes { get { return m_Classes; } }

    /// <summary>
    /// Classify object
    /// </summary>
    public abstract Class Classify(Point x);

    public MaskHandle ApplySampleMask(SampleMaskDelegate mask)
    {
      m_MaskHandle = new MaskHandle(this, mask);
      return m_MaskHandle;
    }

    /// <summary>
    /// Returns all errors of the given algorithm on some initially classified sample
    /// </summary>
    public IEnumerable<Error> GetErrors(ClassifiedSample classifiedSample)
    {
      var errors = new List<Error>();

      foreach (var pData in classifiedSample)
      {
        var res = this.Classify(pData.Key);
        if (res != pData.Value)
          errors.Add(new Error(pData.Key, pData.Value, res));
      }

      return errors;
    }
  }
}
