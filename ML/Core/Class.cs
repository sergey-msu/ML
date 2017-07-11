using System;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Represents a classification class
  /// </summary>
  public struct Class : INamed
  {
    /// <summary>
    /// Unknown (unclassified) class singleton
    /// </summary>
    public static readonly Class Unknown = new Class(-1);

    private readonly string m_Name;
    private readonly double m_Value;
    private readonly bool m_IsUnknown;

    private Class(int value)
    {
      m_Name = "[NONE]";
      m_Value = value;
      m_IsUnknown = true;
    }

    public Class(string name, double? value = null)
    {
      if (string.IsNullOrWhiteSpace(name))
        throw new MLException("Class.ctor(name=null|empty)");

      m_Name = name;
      m_Value = value ?? 0.0F;
      m_IsUnknown = false;
    }

    /// <summary>
    /// Class Name
    /// </summary>
    public string Name { get { return m_Name; } }

    /// <summary>
    /// Some associated value (e.g. {-1, +1} for two-classes classification)
    /// </summary>
    public double Value { get { return m_Value; } }

    /// <summary>
    /// Determines whether the class is unknown class
    /// </summary>
    public bool IsUnknown { get { return m_IsUnknown; } }

    #region Overrides

    public override bool Equals(object obj)
    {
      if (obj==null) return false;
      if (!(obj is Class)) return false;
      if (m_IsUnknown) return false;

      var other = (Class)obj;
      return (!other.m_IsUnknown &&
              m_Name.Equals(other.m_Name) &&
              m_Value == other.m_Value);
    }

    public override int GetHashCode()
    {
      return m_Name.GetHashCode() ^ m_Value.GetHashCode();
    }

    #endregion
  }
}
