using System;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Represents a Feature of some physical object
  /// </summary>
  public class Feature : INamed
  {
    public readonly string m_Name;

    public Feature(string name)
    {
      if (string.IsNullOrWhiteSpace(name))
        throw new MLException("Feature.ctor(name=null|empty)");

      m_Name = name;
    }

    /// <summary>
    /// Feature name
    /// </summary>
    public string Name  { get { return m_Name; } }

    #region Overrides

    public override bool Equals(object obj)
    {
      if (obj==null) return false;

      var other = obj as Feature;
      if (other==null) return false;

      if (!m_Name.Equals(other.m_Name)) return false;

      return true;
    }

    public override int GetHashCode()
    {
      return m_Name.GetHashCode();
    }

    #endregion
  }
}
