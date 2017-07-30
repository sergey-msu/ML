using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core.Logical
{
  /// <summary>
  /// V.I. Donskoy Index
  /// </summary>
  public class DonskoyIndex<TObj> : IndexBase<TObj>
  {
    public override string Name { get {  return "DON"; } }


    public override double Calculate(Predicate<TObj> pattern, ClassifiedSample<TObj> sample)
    {
      int result = 0;
      var array = sample.ToArray();
      var len = array.Length;

      for (int i=0; i<len; i++)
      for (int j=0; j<i; j++)
      {
        var ai = array[i];
        var aj = array[j];

        if (pattern(ai.Key) != pattern(aj.Key) && !ai.Value.Equals(aj.Value))
            result++;
      }

      return result;
    }
  }
}
