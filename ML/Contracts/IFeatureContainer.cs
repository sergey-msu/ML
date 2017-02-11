namespace ML.Contracts
{
  /// <summary>
  /// Contract for physical object that contains some features
  /// </summary>
  public interface IFeaturable<TVal>
  {
    /// <summary>
    /// Gets/sets feature value by its index
    /// </summary>
    TVal this[int idx] { get; set; }

    /// <summary>
    /// Returns raw data array of feature values
    /// </summary>
    TVal[] RawData { get; }

    /// <summary>
    /// Data dimension
    /// </summary>
    int Dimension { get; }
  }
}
