using System;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Stats
{
  /// <summary>
  /// General statistic utils
  /// </summary>
  public static class General
  {
    #region Inner

    /// <summary>
    /// Represents classification error
    /// </summary>
    public class Error
    {
      public Error(Point point, Class realClass, Class calcClass)
      {
        Point = point;
        RealClass = realClass;
        CalcClass = calcClass;
      }

      /// <summary>
      /// Classified point
      /// </summary>
      public readonly Point Point;

      /// <summary>
      /// Real point class
      /// </summary>
      public readonly Class RealClass;

      /// <summary>
      /// Calculated point class
      /// </summary>
      public readonly Class CalcClass;
    }

    #endregion

    /// <summary>
    /// Returns all errors of the given algorithm on some initially classified sample
    /// </summary>
    /// <returns></returns>
    public static IEnumerable<Error> GetErrors(IAlgorithm algorithm, ClassifiedSample classifiedSample)
    {
      var errors = new List<Error>();

      foreach (var pData in classifiedSample)
      {
        var res = algorithm.Classify(pData.Key);
        if (res != pData.Value)
          errors.Add(new Error(pData.Key, pData.Value, res));
      }

      return errors;
    }
  }
}
