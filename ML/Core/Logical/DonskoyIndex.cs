using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core.Logical
{
  /// <summary>
  /// V.I. Donskoy Index
  /// </summary>
  public class DonskoyIndex : IndexBase
  {
    public override string ID { get {  return "DONS"; } }
    public override string Name { get { return "Donskoy Index"; } }

    public override double Calculate(Predicate<double[]> pattern, ClassifiedSample sample)
    {
      int result = 0;
      var prevs = new List<KeyValuePair<double[], Class>>();

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
