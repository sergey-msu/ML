using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Contracts;

namespace ML.Core.Stats
{
  public static class General
  {
    #region Inner

    public class Error
    {
      public Error(Point point, Class realClass, Class calcClass)
      {
        Point = point;
        RealClass = realClass;
        CalcClass = calcClass;
      }

      public readonly Point Point;
      public readonly Class RealClass;
      public readonly Class CalcClass;
    }

    #endregion

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
