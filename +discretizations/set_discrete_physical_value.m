%
% superclass for all sets of discrete physical values
%
% author: Martin F. Schiffner
% date: 2019-02-07
% modified: 2019-02-08
%
classdef set_discrete_physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        S ( 1, : ) physical_values.physical_value	% set consists of multiple physical values

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_physical_value( input )

            % return if no argument
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array
            if ~iscell( input )
                input = cell( input );
            end

            %--------------------------------------------------------------
            % 2.) create sets of discrete physical values
            %--------------------------------------------------------------
            % create column vector of objects
            objects = repmat( objects, size( input ) );

            % set independent and dependent properties
            for index_object = 1:numel( input )

                % ensure row vectors
                if ~isrow( input{ index_object } )
                    errorStruct.message     = sprintf( 'The content of input{ %d } must be a row vector!', index_object );
                    errorStruct.identifier	= 'setting:NoRowVector';
                    error( errorStruct );
                end

                % assign discrete physical values
                objects( index_object ).S = input{ index_object };

            end

        end % function objects = set_discrete_physical_value( input )

        %------------------------------------------------------------------
        % cardinality (overload abs function)
        %------------------------------------------------------------------
        function results = abs( objects )

            % initialize results with zeros
            results = zeros( size( objects ) );

            % extract number of elements
            for index_object = 1:numel( objects )
                results( index_object ) = numel( objects( index_object ).S );
            end

        end % function results = abs( objects )

	end % methods

end % classdef set_discrete_physical_value
