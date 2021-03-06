﻿using System;
using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Contract of ML algorithm (supervised or not, classification or regression etc.)
  /// </summary>
  public interface IAlgorithm : INamed, IMLSerializable
  {
  }

  /// <summary>
  /// Contract of supervised algorithm with training sample
  /// </summary>
  public interface ISupervisedAlgorithm<TSample, TObj, TMark> : IAlgorithm
    where TSample: MarkedSample<TObj, TMark>
  {
    /// <summary>
    /// Initial classified training sample
    /// </summary>
    TSample TrainingSample { get; }

    /// <summary>
    /// Train the algorithm with some initial marked data
    /// </summary>
    void Train(TSample trainingSample);

    /// <summary>
    /// Make a prediction
    /// </summary>
    TMark Predict(TObj obj);

    /// <summary>
    /// Returns all errors of the algorithm on some test classified sample
    /// </summary>
    IEnumerable<ErrorInfo<TObj, TMark>> GetErrors(TSample testSample, double threshold, bool parallel);
  }

  /// <summary>
  /// Contract of supervised algorithm for classification purposes
  /// </summary>
  public interface IClassificationAlgorithm<TObj> : ISupervisedAlgorithm<ClassifiedSample<TObj>, TObj, Class>
  {
    /// <summary>
    /// Returns set of marks (i.e. tags) for the specified object
    /// </summary>
    ClassScore[] PredictTokens(TObj obj, int cnt);
  }

  /// <summary>
  /// Contract of supervised algorithm for regression purposes
  /// </summary>
  public interface IRegressionAlgorithm<TObj> : ISupervisedAlgorithm<RegressionSample<TObj>, TObj, double>
  {
  }

  /// <summary>
  /// Contract of supervised algorithm for multidimensional regression purposes
  /// </summary>
  public interface IMultiRegressionAlgorithm<TObj> : ISupervisedAlgorithm<MultiRegressionSample<TObj>, TObj, double[]>
  {
  }

  /// <summary>
  /// Contract for ML algorithm that can be represented in the form
  /// a(x) = argmax Γ(x,y),
  /// where Γ(x,y) is the proximity of object a to the class y
  /// </summary>
  public interface IGammaAlgorithm<TObj> : IClassificationAlgorithm<TObj>
  {
    /// <summary>
    /// Estimate value of Γ(obj, cls) - object proximity to some class
    /// </summary>
    double CalculateClassScore(TObj obj, Class cls);
  }

  /// <summary>
  /// Contract for general metric classification algorithm
  /// </summary>
  public interface IMetricAlgorithm<TObj> : IGammaAlgorithm<TObj>
  {
    /// <summary>
    /// Space metric
    /// </summary>
    IMetric<TObj> Metric { get; }
  }

  /// <summary>
  /// Contract for general ML classification algorithm with some kernel function
  /// </summary>
  public interface IKernelAlgorithm<TObj> : IGammaAlgorithm<TObj>
  {
    /// <summary>
    /// Kernel
    /// </summary>
    IKernel Kernel { get; }

    /// <summary>
    /// Window width
    /// </summary>
    double H { get; set; }
  }

  /// <summary>
  /// Represents classification error
  /// </summary>
  public class ErrorInfo<TObj, TMark>
  {
    public ErrorInfo(TObj obj, TMark realMark, TMark predMark)
    {
      Object = obj;
      RealMark = realMark;
      PredictedMark = predMark;
    }

    /// <summary>
    /// Classified object
    /// </summary>
    public readonly TObj Object;

    /// <summary>
    /// Real point class
    /// </summary>
    public readonly TMark RealMark;

    /// <summary>
    /// Predicted object class
    /// </summary>
    public readonly TMark PredictedMark;
  }
}
