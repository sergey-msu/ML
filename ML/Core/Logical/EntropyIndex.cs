using System;
using System.Collections.Generic;
using ML.Utils;

namespace ML.Core.Logical
{
  /// <summary>
  /// Multiclass entropy index
  /// </summary>
  public class EntropyIndex<TObj> : IndexBase<TObj>
  {
    #region Inner

      private class ClassBag
      {
        public double Pc { get; set; }
        public double pc { get; set; }
      }

    #endregion

    public override string Name   { get {  return "ENTR"; } }


    public override double Calculate(Predicate<TObj> pattern, ClassifiedSample<TObj> sample)
    {
      var clsInfos = new Dictionary<Class, ClassBag>();
      int p = 0;
      var l = sample.Count;

      foreach (var pData in sample)
      {
        var cls = pData.Value;
        ClassBag bag;
        if (!clsInfos.TryGetValue(cls, out bag))
        {
          bag = new ClassBag();
          clsInfos.Add(cls, bag);
        }

        bag.Pc += 1;
        if (pattern(pData.Key))
        {
          p += 1;
          bag.pc += 1;
        }
      }

      var result = 0.0D;

      foreach (var cData in clsInfos)
      {
        var bag = cData.Value;
        result += GeneralUtils.EntropyH(bag.Pc / l);
        result += (p == 0) ? 0 : -(double)p / l * GeneralUtils.EntropyH(bag.pc / p);
        result += (p == l) ? 0 : -(double)(l - p) / l * GeneralUtils.EntropyH((bag.Pc-bag.pc) / (l - p));
      }

      return result;
    }
  }
}
