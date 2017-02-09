using System;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Represents a classification class
  /// </summary>
  public class Class : INamed
  {
    /// <summary>
    /// Default class singleton
    /// </summary>
    public static readonly Class None = new Class("[NONE]", -1);

    private readonly string m_Name;
    private readonly double  m_Value;
    private readonly int m_Order;

    public Class(string name, int order, double? value = null)
    {
      if (string.IsNullOrWhiteSpace(name))
        throw new MLException("Class.ctor(name=null|empty)");

      m_Name = name;
      m_Order = order;
      m_Value = value ?? 0.0F;
    }

    /// <summary>
    /// Class Name
    /// </summary>
    public string Name { get { return m_Name; } }

    /// <summary>
    /// Class Order
    /// </summary>
    public int Order { get { return m_Order; } }

    /// <summary>
    /// Some associated value (e.g. {-1, +1} for two-classes classification)
    /// </summary>
    public double Value { get { return m_Value; } }

    #region Overrides

    public override bool Equals(object obj)
    {
      if (obj==null) return false;

      var other = obj as Class;
      if (other==null) return false;

      if (!m_Name.Equals(other.m_Name)) return false;

      return m_Value == other.m_Value;
    }

    public override int GetHashCode()
    {
      return m_Name.GetHashCode() ^ m_Value.GetHashCode();
    }

    #endregion
  }
}
