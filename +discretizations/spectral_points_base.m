%
% superclass for all spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-03-09
% modified: 2019-03-11
%
classdef spectral_points_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active %( 1, : ) double { mustBeInteger, mustBeFinite }      % indices of active array elements (1)
        transfer_functions

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points_base( indices_active, transfer_functions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % ensure cell array for transfer_functions
            if ~iscell( transfer_functions )
                transfer_functions = { transfer_functions };
            end

            % ensure equal number of dimensions and sizes of cell arrays
            auxiliary.mustBeEqualSize( indices_active, transfer_functions );

            %--------------------------------------------------------------
            % 2.) create objects
            %--------------------------------------------------------------
            objects = repmat( objects, size( indices_active ) );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( indices_active )

                % ensure equal number of dimensions and sizes of cell array contents
                auxiliary.mustBeEqualSize( indices_active{ index_object }, transfer_functions{ index_object } );

                % ensure row vectors
                if ~isrow( indices_active{ index_object } )
                    errorStruct.message     = sprintf( 'The contents of indices_active{ %d } and transfer_functions{ %d } must be row vectors!', index_object, index_object );
                    errorStruct.identifier	= 'spectral_points_base:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).indices_active = indices_active{ index_object };
                objects( index_object ).transfer_functions = transfer_functions{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = spectral_points_base( indices_active, transfer_functions )

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( object )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( object );

        end % function str_hash = hash( object )

	end % methods

end % classdef spectral_points_base
