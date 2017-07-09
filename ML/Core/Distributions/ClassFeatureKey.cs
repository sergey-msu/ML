using System;

namespace ML.Core.Distributions
{
  public struct ClassFeatureKey
  {
    public ClassFeatureKey(Class cls, int fIdx)
    {
      Class = cls;
      FeatureIdx = fIdx;
    }

    public Class Class;
    public int FeatureIdx;
  }
}
