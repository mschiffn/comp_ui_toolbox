%
% superclass for all spatiospectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-02-25
%
classdef spatiospectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) discretizations.spatial        % spatial discretization
        spectral %( :, : ) discretizations.spectral      % spectral discretization

        % dependent properties
%         set_f_unique ( 1, 1 ) discretizations.set_discrete_frequency

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatiospectral( spatial, spectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

            % ensure cell array
            if ~iscell( spectral )
                spectral = { spectral };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial, spectral );
            % assertion: spatial and spectral have equal sizes

            %--------------------------------------------------------------
            % 2.) create spatiospectral discretizations
            %--------------------------------------------------------------
            objects = repmat( objects, size( spatial ) );

            % set independent and dependent properties
            for index_object = 1:numel( spatial )

                % set independent properties
                objects( index_object ).spatial = spatial( index_object );
                objects( index_object ).spectral = spectral{ index_object };

                % set dependent properties
%                 objects( index_object ).set_f_unique = union( [ objects( index_object ).spectral.set_f_unique ] );

            end % for index_object = 1:numel( spatial )

        end % function objects = spatiospectral( spatial, spectral )

	end % methods

end % classdef spatiospectral
