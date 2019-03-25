%
% superclass for all transducer array parameters
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-03-25
%
classdef parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_elements_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 1 ]	% numbers of elements along each coordinate axis (1)
        str_model = 'Default Array'         % model name
        str_vendor = 'Default Corporation'	% vendor name

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters( N_elements_axis, str_model, str_vendor )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure cell array for N_elements_axis
            if ~iscell( N_elements_axis )
                N_elements_axis = { N_elements_axis };
            end

            % ensure cell array for str_model
            if ~iscell( str_model )
                str_model = { str_model };
            end

            % ensure cell array for str_name
            if ~iscell( str_vendor )
                str_vendor = { str_vendor };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_elements_axis, str_model, str_vendor );

            %--------------------------------------------------------------
            % 2.) create transducer array parameters
            %--------------------------------------------------------------
            objects = repmat( objects, size( N_elements_axis ) );

            % set independent properties
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_elements_axis = N_elements_axis{ index_object };
                objects( index_object ).str_model = str_model{ index_object };
                objects( index_object ).str_vendor = str_vendor{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = parameters( N_elements_axis, str_model, str_vendor )

        %------------------------------------------------------------------
        % project
        %------------------------------------------------------------------
        function parameters = project( parameters, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure positive integers
            mustBeInteger( N_dimensions );
            mustBePositive( N_dimensions );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( parameters, N_dimensions );

            %--------------------------------------------------------------
            % 2.) project parameters onto lower dimension
            %--------------------------------------------------------------
            for index_object = 1:numel( parameters )

                % consider maximum number of dimensions in parameters
                if N_dimensions( index_object ) > numel( parameters( index_object ).N_elements_axis )
                    errorStruct.message     = sprintf( 'N_dimensions( %d ) exceeds number of specified dimensions in parameters( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'project:LargeNumberDimensions';
                    error( errorStruct );
                end

                % extract relevant numbers of elements
                parameters( index_object ).N_elements_axis = parameters( index_object ).N_elements_axis( 1:N_dimensions( index_object ) );

            end % for index_object = 1:numel( parameters )

        end % function parameters = project( parameters, N_dimensions )

        %------------------------------------------------------------------
        % number of elements
        %------------------------------------------------------------------
        function results = N_elements( parameters )

            % initialize results
            results = zeros( size( parameters ) );

            % compute number of elements
            for index_object = 1:numel( parameters )
                results( index_object ) = prod( parameters( index_object ).N_elements_axis, 2 );
            end

        end % function results = N_elements( parameters )

	end % methods

end % classdef parameters
