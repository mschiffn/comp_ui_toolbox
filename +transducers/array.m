%
% superclass for all transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-20
% modified: 2019-03-25
%
classdef array

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        parameters ( 1, 1 ) transducers.parameters
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2              % number of dimensions (1)

        % dependent properties
        N_elements ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 128              % total number of elements (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array( parameters, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.parameters
            if ~isa( parameters, 'transducers.parameters' )
                errorStruct.message     = 'parameters must be transducers.parameters!';
                errorStruct.identifier	= 'array:NoParameters';
                error( errorStruct );
            end

            % ensure positive integers
            mustBeInteger( N_dimensions );
            mustBePositive( N_dimensions );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( parameters, N_dimensions );

            %--------------------------------------------------------------
            % 2.) create transducer arrays
            %--------------------------------------------------------------
            % repeat objects
            objects = repmat( objects, size( parameters ) );

            % set independent and dependent properties
            for index_object = 1:numel( objects )

                % consider maximum number of dimensions in parameters
                if N_dimensions( index_object ) > numel( parameters( index_object ).N_elements_axis )
                    errorStruct.message     = sprintf( 'N_dimensions( %d ) exceeds number of specified dimensions in parameters( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'array:NoPositiveIntegers';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).N_dimensions = N_dimensions( index_object );
                objects( index_object ).parameters = project( parameters( index_object ), objects( index_object ).N_dimensions );

                % dependent properties
                objects( index_object ).N_elements = N_elements( objects( index_object ).parameters );

            end % for index_object = 1:numel( objects )

        end % function objects = array( parameters, N_dimensions )

	end % methods

end % classdef array
