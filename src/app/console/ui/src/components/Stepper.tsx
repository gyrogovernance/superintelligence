import { useState, useEffect } from 'react';

interface StepperProps {
  value: number;
  min?: number;
  label: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function Stepper({ value, min = 0, label, onChange, disabled = false }: StepperProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(value.toString());

  useEffect(() => {
    if (!isEditing) {
      setInputValue(value.toString());
    }
  }, [value, isEditing]);

  const handleDecrement = () => {
    if (value > min) {
      onChange(value - 1);
    }
  };

  const handleIncrement = () => {
    onChange(value + 1);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    const numValue = parseInt(inputValue, 10);
    if (!isNaN(numValue) && numValue >= min) {
      onChange(numValue);
      setInputValue(numValue.toString());
    } else {
      setInputValue(value.toString());
    }
    setIsEditing(false);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleInputBlur();
    } else if (e.key === 'Escape') {
      setInputValue(value.toString());
      setIsEditing(false);
    }
  };

  const handleDisplayClick = () => {
    setIsEditing(true);
    setInputValue(value.toString());
  };

  if (isEditing) {
    return (
      <div role="group" aria-label={label} className="flex items-center gap-2">
        <button
          type="button"
          className="stepper-btn"
          onClick={handleDecrement}
          disabled={disabled || value <= min}
          aria-label={`Decrease ${label}`}
        >
          -
        </button>
        <input
          type="number"
          min={min}
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          disabled={disabled}
          className="w-16 text-center text-xl font-semibold tabular-nums border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed"
          autoFocus
          aria-label={label}
        />
        <button
          type="button"
          className="stepper-btn"
          onClick={handleIncrement}
          disabled={disabled}
          aria-label={`Increase ${label}`}
        >
          +
        </button>
      </div>
    );
  }

  return (
    <div role="group" aria-label={label} className="flex items-center gap-2">
      <button
        type="button"
        className="stepper-btn"
        onClick={handleDecrement}
        disabled={disabled || value <= min}
        aria-label={`Decrease ${label}`}
      >
        -
      </button>
      <span
        className={`w-12 text-center text-xl font-semibold tabular-nums rounded px-1 transition-colors ${
          disabled 
            ? 'opacity-50 cursor-not-allowed' 
            : 'cursor-text hover:bg-gray-100 dark:hover:bg-gray-800'
        }`}
        onClick={handleDisplayClick}
        role={disabled ? undefined : "button"}
        tabIndex={disabled ? undefined : 0}
        onKeyDown={disabled ? undefined : (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleDisplayClick();
          }
        }}
        aria-label={`Edit ${label}, current value: ${value}`}
      >
        {value}
      </span>
      <button
        type="button"
        className="stepper-btn"
        onClick={handleIncrement}
        disabled={disabled}
        aria-label={`Increase ${label}`}
      >
        +
      </button>
    </div>
  );
}
