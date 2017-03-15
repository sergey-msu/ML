using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Base class for algorithm that accepts whole traing sample
  /// </summary>
  public abstract class AlgorithmBase<TObj> : ISupervisedAlgorithm<TObj>
  {
    #region Inner

      public delegate bool SampleMaskDelegate(object p, Class c, int i);

      public class MaskHandle : IDisposable
      {
        private static readonly SampleMaskDelegate ALL = (p, c, i) => true;

        private AlgorithmBase<TObj> m_Algorithm;
        private SampleMaskDelegate m_Mask;
        private ClassifiedSample<TObj> m_MaskedSample;
        private bool m_Disposed;

        public MaskHandle(AlgorithmBase<TObj> algorithm, SampleMaskDelegate mask)
        {
          if (algorithm==null)
            throw new MLException("MaskHandle.ctor(algorithm=null)");

          m_Algorithm = algorithm;
          m_Mask = mask ?? ALL;
        }

        public ClassifiedSample<TObj> MaskedSample
        {
          get
          {
            if (m_Disposed) throw new ObjectDisposedException("MaskHandle has been disposed");

            if (m_MaskedSample == null)
            {
              var maskedSample = m_Algorithm.m_TrainingSample
                                            .Where((kvp, i) => m_Mask(kvp.Key, kvp.Value, i))
                                            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

              m_MaskedSample = new ClassifiedSample<TObj>(maskedSample);
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
      public class ErrorInfo
      {
        public ErrorInfo(object obj, Class realClass, Class calcClass)
        {
          Object = obj;
          RealClass = realClass;
          CalcClass = calcClass;
        }

        /// <summary>
        /// Classified object
        /// </summary>
        public readonly object Object;

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

    protected readonly ClassifiedSample<TObj> m_TrainingSample;
    protected readonly Dictionary<string, Class> m_Classes;
    private MaskHandle m_MaskHandle;

    protected AlgorithmBase(ClassifiedSample<TObj> trainingSample)
    {
      if (trainingSample == null || !trainingSample.Any())
        throw new MLException("AlrogithmBase.ctor(trainingSample=null|empty)");

      m_TrainingSample = new ClassifiedSample<TObj>(trainingSample);
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
    public ClassifiedSample<TObj> TrainingSample
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
    /// Maps object to corresponding class
    /// </summary>
    public abstract Class Classify(TObj obj);

    public MaskHandle ApplySampleMask(SampleMaskDelegate mask)
    {
      m_MaskHandle = new MaskHandle(this, mask);
      return m_MaskHandle;
    }

    /// <summary>
    /// Returns all errors of the given algorithm on some initially classified sample
    /// </summary>
    public IEnumerable<ErrorInfo> GetErrors(ClassifiedSample<TObj> classifiedSample)
    {
      var errors = new List<ErrorInfo>();

      foreach (var pData in classifiedSample)
      {
        var res = this.Classify(pData.Key);
        if (res != pData.Value)
          errors.Add(new ErrorInfo(pData.Key, pData.Value, res));
      }

      return errors;
    }
  }
}
