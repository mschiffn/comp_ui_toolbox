%
% superclass for all times
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-03-21
%
classdef time < physical_values.physical_value_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = time( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                values = 0;
            end

            % check argument
            mustBeNonnegative( values );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_value_base( values );

        end % function objects = time( values )

        %------------------------------------------------------------------
        % right array division (overload rdivide function)
        %------------------------------------------------------------------
        function objects = rdivide( numerators, denominators )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( numerators, 'physical_values.time' )
                if isnumeric( denominators ) || isa( denominators, 'physical_values.time' )

                    % call right array division in superclass
                    objects = rdivide@physical_values.physical_value( numerators, denominators );
                    return;
                end
                if isa( denominators, 'physical_values.physical_value' )
                    errorStruct.message     = 'Denominators must not be physical_values.physical_value when numerators are physical_values.time!';
                    errorStruct.identifier	= 'rdivide:Arguments';
                    error( errorStruct );
                end
            elseif isa( numerators, 'physical_values.physical_value' ) && isa( denominators, 'physical_values.time' )
                errorStruct.message     = 'Denominators must not be physical_values.physical_value when numerators are physical_values.time!';
                errorStruct.identifier	= 'rdivide:Arguments';
                error( errorStruct );
            end

            % multiple objects in numerator / single object in denominator
            if ~isscalar( numerators ) && isscalar( denominators )
                denominators = repmat( denominators, size( numerators ) );
            end

            % single object in numerator / multiple objects in denominator
            if isscalar( numerators ) && ~isscalar( denominators )
                numerators = repmat( numerators, size( denominators ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( numerators, denominators );
            % assertion: numerators and denominators have equal sizes

            %--------------------------------------------------------------
            % 2.) compute results
            %--------------------------------------------------------------
            % create frequencies
            objects = physical_values.frequency( ones( size( numerators ) ) );

            for index_objects = 1:numel( numerators )
                objects( index_objects ) = physical_values.frequency( numerators( index_objects ) / denominators( index_objects ).value );
            end

        end % function objects = rdivide( numerators, denominators )

	end % methods

end % classdef time < physical_values.physical_value_base
