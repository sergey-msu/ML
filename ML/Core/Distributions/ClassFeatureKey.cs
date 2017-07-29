using System;

namespace ML.Core.Distributions
{
  public struct ClassFeatureKey : IEquatable<ClassFeatureKey>
  {
    public ClassFeatureKey(Class cls, int fIdx)
    {
      Class = cls;
      FeatureIdx = fIdx;
    }

    public Class Class;
    public int FeatureIdx;

    public override bool Equals(object obj)
    {
      if (!(obj is ClassFeatureKey)) return false;
      return this.Equals((ClassFeatureKey)obj);
    }

    public bool Equals(ClassFeatureKey other)
    {
      return this.Class.Equals(other.Class) && this.FeatureIdx==other.FeatureIdx;
    }

    public override int GetHashCode()
    {
      return Class.GetHashCode() ^ FeatureIdx.GetHashCode();
    }
  }
}
