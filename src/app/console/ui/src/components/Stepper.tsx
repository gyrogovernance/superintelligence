import { useState, useEffect } from 'react';

interface StepperProps {
  value: number;
  min?: number;
  label: string;
  onChange: (value: number) => void;
}

export function Stepper({ value, min = 0, label, onChange }: StepperProps) {
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
          disabled={value <= min}
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
          className="w-16 text-center text-xl font-semibold tabular-nums border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400"
          autoFocus
          aria-label={label}
        />
        <button
          type="button"
          className="stepper-btn"
          onClick={handleIncrement}
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
        disabled={value <= min}
        aria-label={`Decrease ${label}`}
      >
        -
      </button>
      <span
        className="w-12 text-center text-xl font-semibold tabular-nums cursor-text hover:bg-gray-100 dark:hover:bg-gray-800 rounded px-1 transition-colors"
        onClick={handleDisplayClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
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
        aria-label={`Increase ${label}`}
      >
        +
      </button>
    </div>
  );
}
