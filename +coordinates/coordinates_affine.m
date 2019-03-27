%
% superclass for all affine coordinates
%
% author: Martin F. Schiffner
% date: 2019-03-22
% modified: 2019-03-27
%
classdef coordinates_affine < coordinates.coordinates

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        components ( :, : ) physical_values.length

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coordinates_affine( components )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                components = physical_values.meter( zeros( 1, 2 ) );
            end

            % ensure cell array for components
            if ~iscell( components )
                components = { components };
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@coordinates.coordinates( size( components ) );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure class physical_values.length
                if ~isa( components{ index_object }, 'physical_values.length' )
                    errorStruct.message     = sprintf( 'components{ index_object } must be physical_values.length!', index_object );
                    errorStruct.identifier	= 'coordinates_affine:NoLengths';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).components = components{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = coordinates_affine( components )

    end % methods

end % classdef coordinates_affine < coordinates.coordinates
